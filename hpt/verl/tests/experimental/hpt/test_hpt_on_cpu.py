# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.hpt.algorithm import apply_hpt
from verl.experimental.hpt.dataset import _target_to_text
from verl.experimental.hpt.grpo import compute_policy_loss_grpo_compat
from verl.experimental.hpt.loss import compute_policy_loss_hpt
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [10 + len(text), 11]}


class _ActorConfig:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    clip_ratio_c = 3.0
    global_batch_info = {}

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
        },
    )


def test_target_normalization():
    assert _target_to_text("answer") == "answer"
    assert _target_to_text({"role": "assistant", "content": "answer"}) == "answer"
    assert _target_to_text([{"role": "assistant", "content": "answer"}]) == "answer"
    assert _target_to_text(np.array([{"role": "assistant", "content": "answer"}], dtype=object)) == "answer"


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


def test_switch_replaces_failed_group_and_masks_dp_padding():
    reward = torch.zeros(4, 3)
    reward[2, 1] = 1
    output, output_reward, metrics = apply_hpt(
        _batch(),
        reward,
        _Tokenizer(),
        {"strategy": "switch", "switch_gate": 0},
        dp_size=2,
    )

    # Failed group a -> one SFT row. Solved group b -> two RL rows. One
    # additional row is padding so the actor DP size remains valid.
    assert len(output) == 4
    assert output.batch["hpt_sft_mask"].tolist() == [True, False, False, True]
    assert output.batch["hpt_valid_mask"].tolist() == [True, True, True, False]
    assert output_reward.shape == (4, 3)
    assert metrics["hpt/sft_groups"] == 1
    assert metrics["hpt/rl_groups"] == 1


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
