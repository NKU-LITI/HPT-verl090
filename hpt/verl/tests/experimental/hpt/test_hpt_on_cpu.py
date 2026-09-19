# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.hpt.actor_update import (
    legacy_grad_norm_skip,
    legacy_gradient_accumulation,
    scale_legacy_backward_loss,
)
from verl.experimental.hpt.algorithm import apply_hpt
from verl.experimental.hpt.dataset import _strip_duplicate_think_prefix, _target_to_text
from verl.experimental.hpt.grpo import compute_policy_loss_grpo_compat
from verl.experimental.hpt.layout import balance_hpt_batch_like_reference, pad_hpt_batch_like_reference
from verl.experimental.hpt.loss import compute_hpt_legacy_entropy_loss, compute_policy_loss_hpt
from verl.experimental.hpt.reward import compute_score as compute_hpt_reward
from verl.experimental.hpt.reward import legacy_reward_input
from verl.experimental.hpt.reward_manager import decode_legacy_response
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _attach_hpt_reward,
    _global_ppo_mini_batch_size,
    _hpt_legacy_gradient_accumulation,
)
from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import rearrange_micro_batches_hpt_legacy
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.engine_workers import _select_valid_rows


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [10 + len(text), 11]}

    def decode(self, token_ids, skip_special_tokens=False):
        del token_ids
        return "reasoning \\boxed{1}" + ("" if skip_special_tokens else "<eos>")


class _ActorConfig:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    clip_ratio_c = 3.0
    global_batch_info = {}

    def __init__(self, *, remove_clip=True, remove_token_mean=True):
        self.policy_loss = _PolicyConfig(
            remove_clip=remove_clip,
            remove_token_mean=remove_token_mean,
        )
        self.global_batch_info = {}

    def get(self, key, default=None):
        return getattr(self, key, default)


class _PolicyConfig:
    def __init__(self, *, remove_clip, remove_token_mean):
        self.loss_remove_clip = remove_clip
        self.loss_remove_token_mean = remove_token_mean

    def get(self, key, default=None):
        return getattr(self, key, default)


class _GRPOActorConfig(_ActorConfig):
    def __init__(self, *, remove_clip, remove_token_mean):
        self.policy_loss = _PolicyConfig(
            remove_clip=remove_clip,
            remove_token_mean=remove_token_mean,
        )
        self.global_batch_info = {}


def _reference_hpt_fixed_microbatch(
    *,
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    prefix_mask,
    offline_mode,
    entropy,
    entropy_coeff=0.001,
    sft_loss_coef=1.0,
    off_policy_beta=0.1,
):
    """Executable transcription of the canonical 0.1 HPT micro-batch objective."""
    response_mask = response_mask.bool()
    prefix_mask = prefix_mask.bool()
    response_width = response_mask.shape[-1]
    target_rows = prefix_mask.any(dim=-1)
    on_token_loss = -advantages * torch.exp(log_prob - old_log_prob)
    zero = log_prob.sum() * 0.0

    if offline_mode == 0:
        on_mask = response_mask & (~target_rows).unsqueeze(-1)
        on_loss = (on_token_loss * on_mask).sum() / response_width
        target_mask = response_mask & target_rows.unsqueeze(-1)
        sft_loss = (
            (-log_prob * target_mask).sum() / target_mask.sum() * sft_loss_coef if target_mask.any() else zero
        )
        off_loss = zero
    else:
        probability = torch.exp(log_prob)
        off_token_loss = -advantages * probability / (probability + off_policy_beta)
        on_mask = response_mask & ~prefix_mask
        off_mask = response_mask & prefix_mask
        on_loss = (on_token_loss * on_mask).sum() / response_width
        off_loss = (off_token_loss * off_mask).sum() / response_width
        sft_loss = zero

    pg_loss = on_loss + sft_loss + off_loss
    entropy_loss = (entropy * response_mask).sum() / response_mask.sum()
    total_loss = pg_loss - entropy_coeff * entropy_loss
    return {
        "on_policy_loss": on_loss,
        "sft_loss": sft_loss,
        "off_policy_loss": off_loss,
        "pg_loss": pg_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss,
    }


def _hpt_objective_case(case):
    log_prob = case["log_prob"].clone().requires_grad_()
    config = _ActorConfig()
    # A deliberately unrelated global denominator proves that the HPT legacy
    # fixed-microbatch objective does not use verl 0.9 global aggregation.
    config.global_batch_info = {
        "dp_size": 1,
        "batch_num_tokens": int(case["response_mask"].sum()) + 17,
        "global_batch_size": case["response_mask"].shape[0] + 11,
        "loss_scale_factor": None,
    }
    target_rows = case["prefix_mask"].any(dim=-1)
    mode = case["offline_mode"]
    pg_loss, metrics = compute_policy_loss_hpt(
        old_log_prob=case["old_log_prob"],
        log_prob=log_prob,
        advantages=case["advantages"],
        response_mask=case["response_mask"],
        config=config,
        hpt_sft_mask=target_rows,
        hpt_valid_mask=torch.ones_like(target_rows),
        hpt_sft_weight=torch.full_like(target_rows, case["sft_loss_coef"], dtype=torch.float32),
        hpt_offline_mode=torch.full_like(target_rows, mode, dtype=torch.int8),
        hpt_off_policy_beta=torch.full_like(target_rows, case["off_policy_beta"], dtype=torch.float32),
    )
    entropy_loss = compute_hpt_legacy_entropy_loss(case["entropy"], case["response_mask"])
    total_loss = pg_loss - case["entropy_coeff"] * entropy_loss
    return log_prob, {
        "on_policy_loss": pg_loss.new_tensor(metrics["hpt/rl_loss"]),
        "sft_loss": pg_loss.new_tensor(metrics["hpt/sft_loss"]),
        "off_policy_loss": pg_loss.new_tensor(metrics["hpt/off_policy_loss"]),
        "pg_loss": pg_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss,
    }


def _fixed_objective_cases():
    def case(name, log_prob, old_log_prob, advantages, response_mask, prefix_mask, offline_mode):
        log_prob = torch.tensor(log_prob, dtype=torch.float64)
        return {
            "name": name,
            "log_prob": log_prob,
            "old_log_prob": torch.tensor(old_log_prob, dtype=torch.float64),
            "advantages": torch.tensor(advantages, dtype=torch.float64),
            "response_mask": torch.tensor(response_mask, dtype=torch.bool),
            "prefix_mask": torch.tensor(prefix_mask, dtype=torch.bool),
            "whether_off": torch.tensor([offline_mode == 1] * len(log_prob), dtype=torch.bool),
            "offline_mode": offline_mode,
            "entropy": torch.arange(1, log_prob.numel() + 1, dtype=torch.float64).reshape_as(log_prob) / 10,
            "entropy_coeff": 0.001,
            "sft_loss_coef": 0.7,
            "off_policy_beta": 0.1,
        }

    return [
        case(
            "A_pure_on_policy",
            [[-0.2, -0.4, -0.8], [-1.1, -0.7, -2.0]],
            [[-0.21, -0.39, -0.8], [-1.12, -0.72, -2.0]],
            [[1.0, -0.5, 0.0], [-1.0, 0.3, 0.0]],
            [[1, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 0]],
            0,
        ),
        case(
            "B_pure_sft",
            [[-1.0, -2.0, -4.0], [-0.4, -0.8, -1.2]],
            [[-1.0, -2.0, -4.0], [-0.4, -0.8, -1.2]],
            [[0, 0, 0], [0, 0, 0]],
            [[1, 1, 0], [1, 1, 1]],
            [[1, 1, 0], [1, 1, 1]],
            0,
        ),
        case(
            "C_mixed_rl_sft",
            [[-0.1, -0.3, -2.0], [-1.0, -1.5, -3.0], [-0.7, -2.0, -2.0]],
            [[-0.2, -0.2, -2.0], [-1.0, -1.5, -3.0], [-0.8, -2.0, -2.0]],
            [[1.0, -1.0, 0], [0, 0, 0], [-0.5, 0, 0]],
            [[1, 1, 0], [1, 1, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
            0,
        ),
        case(
            "D_pure_off_policy",
            [[-0.4, -1.0, -2.0], [-0.7, -1.4, -2.1]],
            [[-0.5, -1.1, -2.0], [-0.8, -1.3, -2.1]],
            [[1.0, -0.5, 0], [-1.0, 0.25, 0]],
            [[1, 1, 0], [1, 1, 0]],
            [[1, 1, 0], [1, 1, 0]],
            1,
        ),
        case(
            "E_mixed_on_off",
            [[-0.2, -0.9, -2.0], [-0.5, -1.2, -2.0]],
            [[-0.25, -0.8, -2.0], [-0.55, -1.1, -2.0]],
            [[1.0, -0.5, 0], [-1.0, 0.75, 0]],
            [[1, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0]],
            1,
        ),
        case(
            "F_variable_lengths",
            [[-0.3, -0.6, -0.9, -1.2], [-0.4, -0.8, -1.6, -2.0], [-0.2, -2.0, -2.0, -2.0]],
            [[-0.35, -0.65, -0.95, -1.2], [-0.45, -0.75, -1.5, -2.0], [-0.25, -2.0, -2.0, -2.0]],
            [[1, 1, -1, 0], [-1, 0.5, 0.25, 0], [0.5, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
            1,
        ),
        case(
            "G_ratio_near_one",
            [[-0.500001, -0.999999], [-1.500002, -0.299998]],
            [[-0.5, -1.0], [-1.5, -0.3]],
            [[1, -1], [0.5, -0.25]],
            [[1, 1], [1, 1]],
            [[0, 0], [0, 0]],
            0,
        ),
        case(
            "H_ratio_outside_clip",
            [[1.0, -4.0, -0.2], [-3.0, 0.8, -2.0]],
            [[-1.0, -1.0, -0.2], [-1.0, -1.0, -2.0]],
            [[1.0, 1.0, 0], [-1.0, -1.0, 0]],
            [[1, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 0]],
            0,
        ),
        case(
            "I_mixed_advantage_signs",
            [[-0.2, -0.7, -1.4], [-0.5, -1.0, -1.8], [-0.1, -0.9, -2.0]],
            [[-0.3, -0.6, -1.3], [-0.4, -1.1, -1.7], [-0.2, -0.8, -2.0]],
            [[2.0, -1.0, 0.5], [-2.0, 1.5, -0.25], [0.75, -0.5, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            1,
        ),
    ]


def test_hpt_fixed_microbatch_objective_and_logprob_gradient_match_reference():
    component_names = ("on_policy_loss", "sft_loss", "off_policy_loss", "pg_loss", "entropy_loss", "total_loss")
    for case in _fixed_objective_cases():
        reference_log_prob = case["log_prob"].clone().requires_grad_()
        reference = _reference_hpt_fixed_microbatch(
            old_log_prob=case["old_log_prob"],
            log_prob=reference_log_prob,
            advantages=case["advantages"],
            response_mask=case["response_mask"],
            prefix_mask=case["prefix_mask"],
            offline_mode=case["offline_mode"],
            entropy=case["entropy"],
            entropy_coeff=case["entropy_coeff"],
            sft_loss_coef=case["sft_loss_coef"],
            off_policy_beta=case["off_policy_beta"],
        )
        reference["total_loss"].backward()

        migrated_log_prob, migrated = _hpt_objective_case(case)
        migrated["total_loss"].backward()

        for component in component_names:
            torch.testing.assert_close(migrated[component], reference[component], rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(migrated_log_prob.grad, reference_log_prob.grad, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(migrated_log_prob.grad.norm(), reference_log_prob.grad.norm(), rtol=1e-7, atol=1e-9)
        cosine = torch.nn.functional.cosine_similarity(
            migrated_log_prob.grad.flatten(), reference_log_prob.grad.flatten(), dim=0
        )
        torch.testing.assert_close(cosine, cosine.new_tensor(1.0), rtol=1e-7, atol=1e-9)


def _batch():
    prompts = torch.tensor([[0, 3, 4], [0, 3, 4], [0, 5, 6], [0, 5, 6]])
    responses = torch.tensor([[7, 2, 0], [8, 2, 0], [9, 2, 0], [7, 2, 0]])
    response_mask = torch.tensor([[1, 1, 0]] * 4)
    attention_mask = torch.cat([(prompts != 0).long(), response_mask], dim=1)
    tensor_batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": attention_mask,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "position_ids": torch.clip(attention_mask.cumsum(-1) - 1, min=0),
        },
        batch_size=[4],
    )
    return DataProto(
        batch=tensor_batch,
        non_tensor_batch={
            "uid": np.array(["a", "a", "b", "b"], dtype=object),
            "hpt_target": np.array(["A", "A", "B", "B"], dtype=object),
            "data_source": np.array(["math"] * 4, dtype=object),
            "reward_model": np.array([{"ground_truth": "1"}] * 4, dtype=object),
        },
    )


def test_target_normalization():
    assert _target_to_text("answer") == "answer"
    assert _target_to_text({"role": "assistant", "content": "answer"}) == "answer"
    assert _target_to_text([{"role": "assistant", "content": "answer"}]) == "answer"
    assert _target_to_text(np.array([{"role": "assistant", "content": "answer"}], dtype=object)) == "answer"
    assert _strip_duplicate_think_prefix("prompt<think>\n", "<think>\nanswer") == "answer"
    assert _strip_duplicate_think_prefix("prompt", "<think>\nanswer") == "<think>\nanswer"


def test_hpt_reward_uses_legacy_entropy_math_and_text_prefix():
    assert legacy_reward_input("reasoning \\boxed{1}<eos>") == "<think>\nreasoning \\boxed{1}<eos>"
    assert compute_hpt_reward("math", "reasoning \\boxed{1}<eos>", "1") == 1.0
    assert compute_hpt_reward("math", "reasoning \\boxed{2}<eos>", "1") == 0.0
    assert compute_hpt_reward("math", "answer is 1", "1") == 0.0
    assert decode_legacy_response(_Tokenizer(), torch.tensor([7, 2])) == "reasoning \\boxed{1}<eos>"


def test_hpt_reward_is_compatible_with_async_reward_loop_thread_pool():
    # RewardManagerBase.run_single dispatches synchronous scorers through
    # run_in_executor; legacy SIGALRM guards must not make that path unusable.
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(compute_hpt_reward, "math", "reasoning \\boxed{1}<eos>", "1").result()
    assert result == 1.0


def test_gen_batch_keeps_target_on_driver_batch():
    data = DataProto(
        batch=TensorDict({"dummy": torch.ones(1, 1)}, batch_size=[1]),
        non_tensor_batch={
            "raw_prompt": np.array([[{"role": "user", "content": "q"}]], dtype=object),
            "hpt_target": np.array(["answer"], dtype=object),
            "data_source": np.array(["math"], dtype=object),
        },
    )
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    gen_batch = trainer._get_gen_batch(data)
    assert data.non_tensor_batch["hpt_target"].tolist() == ["answer"]
    assert gen_batch.non_tensor_batch["hpt_target"].tolist() == ["answer"]


def test_post_hpt_reward_stays_aligned_through_balance():
    batch = _batch()
    batch.batch["hpt_debug_sentinel"] = torch.arange(len(batch), dtype=torch.long)
    reward = torch.arange(len(batch), dtype=torch.float32).unsqueeze(-1)
    _attach_hpt_reward(batch, reward)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": [0, 1]})
    trainer.use_prefix_grouper = False
    trainer._balance_batch(batch, metrics={})

    row_ids = batch.batch["hpt_debug_sentinel"]
    assert row_ids.tolist() != list(range(len(batch)))
    torch.testing.assert_close(batch.batch["token_level_scores"].squeeze(-1), row_ids.float())


def test_hpt_ppo_mini_batch_target_matches_reference():
    # Reference worker normalization: configured 32 prompts * rollout.n 8,
    # then DP dispatch turns the fixed global 256 into 128 rows per rank.
    assert _global_ppo_mini_batch_size(configured_size=32, rollout_n=8) == 256


def test_update_actor_keeps_hpt_mini_batch_target_independent_of_routed_rows():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"n": 8, "temperature": 1.0, "multi_turn": {"enable": False}},
                "actor": {
                    "ppo_mini_batch_size": 32,
                    "ppo_micro_batch_size": 1,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": False,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                },
            },
            "hpt": {"enabled": True},
        }
    )
    trainer.distillation_config = None
    captured = []

    class _ActorWorkerGroup:
        def update_actor(self, data):
            captured.append(
                {
                    "rows": len(data),
                    "mini_batch_size": tu.get(data, "mini_batch_size"),
                    "valid_mask_key": tu.get(data, "valid_mask_key"),
                    "allow_partial": tu.get(data, "allow_partial_mini_batch"),
                    "legacy_packing": tu.get(data, "hpt_legacy_dynamic_batching"),
                    "gradient_accumulation": tu.get(data, "hpt_legacy_gradient_accumulation"),
                    "max_grad_norm": tu.get(data, "hpt_legacy_max_grad_norm"),
                }
            )
            return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"mfu": 0.0}})

    trainer.actor_rollout_wg = _ActorWorkerGroup()
    for routed_rows in (2, 4):
        batch = _batch().select_idxs(list(range(routed_rows)))
        batch.batch["hpt_valid_mask"] = torch.ones(routed_rows, dtype=torch.bool)
        trainer._update_actor(batch)

    assert captured == [
        {
            "rows": 2,
            "mini_batch_size": 256,
            "valid_mask_key": "hpt_valid_mask",
            "allow_partial": True,
            "legacy_packing": True,
            "gradient_accumulation": 32,
            "max_grad_norm": 80.0,
        },
        {
            "rows": 4,
            "mini_batch_size": 256,
            "valid_mask_key": "hpt_valid_mask",
            "allow_partial": True,
            "legacy_packing": True,
            "gradient_accumulation": 32,
            "max_grad_norm": 80.0,
        },
    ]


def test_hpt_actor_update_legacy_scale_and_skip_boundaries():
    actor_config = OmegaConf.create({"ppo_mini_batch_size": 32, "ppo_micro_batch_size": 1})
    assert _hpt_legacy_gradient_accumulation(actor_config) == 32
    assert legacy_gradient_accumulation(32, 1) == 32

    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    raw_losses = [parameter @ torch.tensor([16.0, 0.0]), parameter @ torch.tensor([0.0, 8.0])]
    for raw_loss in raw_losses:
        scale_legacy_backward_loss(raw_loss, 32).backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([0.5, 0.25]))

    assert legacy_grad_norm_skip(torch.tensor(79.999), 80.0) == (False, False)
    assert legacy_grad_norm_skip(torch.tensor(80.0), 80.0) == (False, False)
    assert legacy_grad_norm_skip(torch.tensor(80.001), 80.0) == (True, False)
    assert legacy_grad_norm_skip(torch.tensor(float("inf")), 80.0) == (True, True)


def test_hpt_fsdp_optimizer_uses_pre_clip_norm_for_whole_step_skip():
    class _CountingSGD(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=0.1)
            self.step_count = 0

        def step(self, closure=None):
            self.step_count += 1
            return super().step(closure)

    def run(pre_clip_gradient):
        parameter = torch.nn.Parameter(torch.tensor([0.0]))
        optimizer = _CountingSGD([parameter])
        parameter.grad = torch.tensor([pre_clip_gradient])
        engine = SimpleNamespace(
            optimizer_config=SimpleNamespace(clip_grad=1.0),
            module=torch.nn.ParameterList([parameter]),
            optimizer=optimizer,
            scaler=None,
            _hpt_legacy_max_grad_norm=80.0,
            _qat_enabled=False,
        )
        grad_norm = FSDPEngine.optimizer_step(engine)
        return grad_norm, optimizer.step_count, parameter.detach().item(), parameter.grad

    grad_norm, step_count, parameter, gradient = run(79.999)
    assert grad_norm == torch.tensor(79.999).item()
    assert step_count == 1
    assert parameter < 0
    assert gradient is not None

    grad_norm, step_count, parameter, gradient = run(80.001)
    assert grad_norm == torch.tensor(80.001).item()
    assert step_count == 0
    assert parameter == 0.0
    assert gradient is None


def test_hpt_padding_is_removed_before_partial_mini_batch_split():
    data = TensorDict(
        {
            "row_id": torch.arange(132),
            "hpt_valid_mask": torch.tensor([True] * 130 + [False] * 2),
        },
        batch_size=[132],
    )
    real_data = _select_valid_rows(data, "hpt_valid_mask")
    mini_batches = list(
        tu.make_iterator(
            real_data,
            mini_batch_size=128,
            epochs=1,
            dataloader_kwargs={"shuffle": False},
            allow_partial=True,
        )
    )

    assert [len(mini_batch) for mini_batch in mini_batches] == [128, 2]
    assert torch.equal(mini_batches[0]["row_id"], torch.arange(128))
    assert torch.equal(mini_batches[1]["row_id"], torch.arange(128, 130))


def test_hpt_padding_selection_supports_async_rollout_nested_tensors():
    nested = torch.nested.as_nested_tensor(
        [torch.tensor([10, 11]), torch.tensor([20]), torch.tensor([30, 31, 32])],
        layout=torch.jagged,
    )
    data = TensorDict(
        {
            "nested_ids": nested,
            "row_id": torch.arange(3),
            "hpt_valid_mask": torch.tensor([True, False, True]),
        },
        batch_size=[3],
    )

    selected = _select_valid_rows(data, "hpt_valid_mask")

    assert selected["row_id"].tolist() == [0, 2]
    assert selected["nested_ids"].is_nested
    assert [row.tolist() for row in selected["nested_ids"].unbind()] == [[10, 11], [30, 31, 32]]


def _partition_batch(lengths, valid_mask=None):
    width = 10
    attention_mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1
    if valid_mask is None:
        valid_mask = torch.ones(len(lengths), dtype=torch.bool)
    return TensorDict(
        {
            "input_ids": torch.arange(width).repeat(len(lengths), 1),
            "attention_mask": attention_mask,
            "row_id": torch.arange(len(lengths)),
            "hpt_valid_mask": valid_mask,
        },
        batch_size=[len(lengths)],
    )


def test_hpt_dynamic_micro_batch_partitions_match_legacy_reference_cases():
    cases = [
        ([5, 5, 5, 5], None, [[0, 3], [1, 2]]),
        ([2, 9, 3, 8, 4], None, [[0, 2, 4], [1], [3]]),
        ([4, 6, 5, 5], None, [[2, 3], [0, 1]]),
        ([5, 6], None, [[1], [0]]),
        ([4, 4, 4], None, [[0, 1], [2]]),
        ([3, 3, 10, 2], None, [[2], [0, 1, 3]]),
        ([6, 9, 4, 5], None, [[2, 3], [1], [0]]),
        ([6, 4, 10, 5], [True, True, False, True], [[1, 2], [0]]),
        ([8, 2, 4, 9, 3], None, [[1, 2, 4], [3], [0]]),
    ]
    for lengths, valid_mask, expected in cases:
        mask = None if valid_mask is None else torch.tensor(valid_mask, dtype=torch.bool)
        batch = _select_valid_rows(_partition_batch(lengths, mask), "hpt_valid_mask")
        _, partitions = rearrange_micro_batches_hpt_legacy(batch, max_token_len=10)
        assert partitions == expected


def test_prepare_micro_batches_selects_legacy_partition_only_when_flagged():
    data = _partition_batch([2, 9, 3, 8, 4])
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=True,
        max_token_len_per_gpu=10,
        sp_size=1,
        hpt_legacy_dynamic_batching=True,
    )
    data = next(
        tu.make_iterator(
            _select_valid_rows(data, "hpt_valid_mask"),
            mini_batch_size=5,
            epochs=1,
            dataloader_kwargs={"shuffle": False},
        )
    )
    _, hpt_partitions = prepare_micro_batches(data, same_micro_num_in_dp=False)
    assert hpt_partitions == [[0, 2, 4], [1], [3]]

    stock_data = data.clone()
    tu.assign_non_tensor(stock_data, hpt_legacy_dynamic_batching=False)
    _, stock_partitions = prepare_micro_batches(stock_data, same_micro_num_in_dp=False)
    assert stock_partitions != hpt_partitions


def test_switch_matches_legacy_post_selector_order_before_padding():
    reward = torch.zeros(4, 3)
    reward[2, 1] = 1
    output, output_reward, metrics = apply_hpt(
        _batch(),
        reward,
        _Tokenizer(),
        {"strategy": "switch", "switch_gate": 0},
        dp_size=2,
    )

    # Reference retains solved rows in original order, then appends failed
    # targets in sorted-UID order. Padding occurs later in the trainer.
    assert len(output) == 3
    assert output.non_tensor_batch["uid"].tolist() == ["b", "b", "a"]
    assert output.batch["hpt_sft_mask"].tolist() == [False, False, True]
    assert output.batch["hpt_valid_mask"].tolist() == [True, True, True]
    assert output.batch["prefix_mask"].tolist() == [[False] * 3, [False] * 3, [True] * 3]
    assert output_reward.sum(dim=-1).tolist() == [1.0, 0.0, 1.0]
    assert output_reward.shape == (3, 3)
    assert metrics["hpt/sft_groups"] == 1
    assert metrics["hpt/rl_groups"] == 1


def test_hpt_final_padding_and_balance_match_legacy_layout():
    costs = [2, 9, 3, 8, 4]
    attention_mask = torch.zeros(5, 9, dtype=torch.long)
    for row, cost in enumerate(costs):
        attention_mask[row, :cost] = 1
    batch = DataProto(
        batch=TensorDict(
            {
                "attention_mask": attention_mask,
                "row_id": torch.arange(5),
                "hpt_valid_mask": torch.ones(5, dtype=torch.bool),
            },
            batch_size=[5],
        ),
        non_tensor_batch={"uid": np.array(["u0", "u1", "u2", "u3", "u4"], dtype=object)},
    )

    batch, pad_size = pad_hpt_batch_like_reference(batch, 8, mark_actor_padding=True)
    assert pad_size == 3
    assert batch.batch["row_id"].tolist() == [0, 1, 2, 3, 4, 4, 4, 4]
    assert batch.non_tensor_batch["uid"].tolist() == ["u0", "u1", "u2", "u3", "u4", "u4", "u4", "u4"]
    assert batch.batch["whether_pad"].tolist() == [False] * 5 + [True] * 3
    assert batch.batch["hpt_valid_mask"].tolist() == [True] * 5 + [False] * 3

    partitions, sequence_costs = balance_hpt_batch_like_reference(batch, dp_size=2)
    assert sequence_costs == [2, 9, 3, 8, 4, 4, 4, 4]
    assert partitions == [[2, 3, 4, 6], [0, 1, 5, 7]]
    assert batch.batch["row_id"].tolist() == [2, 3, 4, 4, 0, 1, 4, 4]
    # Padding participates in balance/rank assignment, then exits before actor mini-batching.
    real_batch = _select_valid_rows(batch.to_tensordict(), "hpt_valid_mask")
    assert real_batch["row_id"].tolist() == [2, 3, 4, 0, 1]


def test_hpt_loss_combines_sft_and_ppo_and_ignores_padding():
    old_log_prob = torch.zeros(3, 2)
    log_prob = torch.tensor([[-1.0, -1.0], [-2.0, -2.0], [-100.0, -100.0]], requires_grad=True)
    advantages = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    response_mask = torch.ones(3, 2)

    loss, metrics = compute_policy_loss_hpt(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_ActorConfig(),
        hpt_sft_mask=torch.tensor([True, False, True]),
        hpt_valid_mask=torch.tensor([True, True, False]),
        hpt_sft_weight=torch.ones(3),
    )

    assert torch.isfinite(loss)
    assert metrics["hpt/sft_loss"] == 1.0
    loss.backward()
    assert log_prob.grad is not None
    assert torch.equal(log_prob.grad[2], torch.zeros(2))


def test_off_policy_and_srft_modes_are_finite():
    for mode in (1, 2):
        old_log_prob = torch.zeros(3, 2)
        log_prob = torch.full((3, 2), -1.0, requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0], [0.5, 0.5], [-0.5, -0.5]])
        response_mask = torch.ones(3, 2)
        loss, metrics = compute_policy_loss_hpt(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=_ActorConfig(),
            hpt_sft_mask=torch.tensor([True, False, False]),
            hpt_valid_mask=torch.ones(3, dtype=torch.bool),
            hpt_sft_weight=torch.ones(3),
            hpt_offline_mode=torch.full((3,), mode),
            hpt_off_policy_beta=torch.full((3,), 0.1),
            hpt_correct_mask=torch.tensor([True, True, False]),
            entropy=torch.ones(3, 2),
        )
        assert torch.isfinite(loss)
        assert metrics["hpt/off_policy_loss"] != 0
        if mode == 2:
            assert metrics["hpt/sft_loss"] != 0
            assert metrics["hpt/srft_on_loss"] != 0
        loss.backward()
        assert torch.isfinite(log_prob.grad).all()


def test_grpo_compat_removes_clip_and_valid_token_mean():
    old_log_prob = torch.zeros(2, 3)
    log_prob = torch.log(torch.tensor([[2.0, 0.5, 4.0], [0.4, 1.5, 2.0]])).requires_grad_()
    advantages = torch.tensor([[1.0, -2.0, 9.0], [0.5, -1.0, 8.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    loss, metrics = compute_policy_loss_grpo_compat(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_GRPOActorConfig(remove_clip=True, remove_token_mean=True),
    )
    expected = (-advantages * torch.exp(log_prob) * response_mask).sum() / response_mask.shape[-1]
    torch.testing.assert_close(loss, expected)
    assert metrics["grpo_compat/remove_clip"] == 1.0
    assert metrics["grpo_compat/remove_token_mean"] == 1.0
    loss.backward()
    assert torch.isfinite(log_prob.grad).all()


def test_grpo_compat_can_restore_clip_and_token_mean():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.log(torch.tensor([[2.0, 0.5]])).requires_grad_()
    advantages = torch.tensor([[1.0, -2.0]])
    response_mask = torch.ones(1, 2)

    loss, _ = compute_policy_loss_grpo_compat(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_GRPOActorConfig(remove_clip=False, remove_token_mean=False),
    )
    ratio = torch.exp(log_prob)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 0.8, 1.2)
    expected = torch.maximum(unclipped, clipped).mean()
    torch.testing.assert_close(loss, expected)
