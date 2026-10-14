# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Driver-side batch transformation for Hybrid Post-Training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.hpt.reward import compute_score as compute_hpt_reward
from verl.experimental.hpt.reward_manager_origin import decode_legacy_response
from verl.utils.model import compute_position_id_with_mask


def _get(config: DictConfig | dict, key: str, default: Any) -> Any:
    return config.get(key, default)


def _reward_matches(scores: torch.Tensor, value: float) -> torch.Tensor:
    """Match sequence rewards using the same tolerance as HPT routing."""
    return torch.isclose(scores, scores.new_tensor(value))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return a row-wise mean while preserving the legacy NaN for an empty side."""
    selected = values[mask.to(device=values.device)]
    if selected.numel() == 0:
        return float("nan")
    return selected.float().mean().detach().item()


def _compute_batch_metrics(
    scores: torch.Tensor,
    uids: np.ndarray,
    *,
    success_reward: float,
    failure_reward: float,
    format_reward: float,
) -> dict[str, float]:
    """Compute the rollout-batch metrics emitted by the 0.1 HPT trainer."""
    solve_none = 0
    solve_none_format = 0
    solve_all = 0
    success_counts = []
    for uid in np.unique(uids):
        group_scores = scores[np.flatnonzero(uids == uid).tolist()]
        success_counts.append(int(_reward_matches(group_scores, success_reward).sum().item()))
        if _reward_matches(group_scores, failure_reward).all():
            solve_none += 1
        elif _reward_matches(group_scores, success_reward).all():
            solve_all += 1
        elif _reward_matches(group_scores, format_reward).all():
            solve_none_format += 1

    metrics = {
        "batch/solve_none": float(solve_none),
        "batch/solve_none_format": float(solve_none_format),
        "batch/solve_all": float(solve_all),
        "batch/solved": _reward_matches(scores, success_reward).float().mean().item(),
        "batch/failed": _reward_matches(scores, failure_reward).float().mean().item(),
    }
    # HPT routes on the group-level best-of-n result, not only on the marginal
    # response reward. Preserve the original scalar names above and expose the
    # complete success-count histogram so a rollout-runtime change that alters
    # within-prompt joint sampling cannot hide behind a similar mean reward.
    if success_counts:
        rollout_n = max(int((uids == uid).sum()) for uid in np.unique(uids))
        for successes in range(rollout_n + 1):
            metrics[f"batch/success_count_{successes}"] = float(success_counts.count(successes))
        success_count_tensor = torch.tensor(success_counts, dtype=torch.float32)
        metrics["batch/success_count_mean"] = float(success_count_tensor.mean().item())
        metrics["batch/success_count_std"] = float(success_count_tensor.std(unbiased=False).item())
    return metrics


def _compute_on_off_metrics(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute post-routing on/off-policy statistics before HPT padding."""
    off_policy_mask = prefix_mask.any(dim=-1)
    on_policy_mask = ~off_policy_mask
    response_length = batch.batch["response_mask"].sum(dim=-1)
    sequence_score = reward_tensor.sum(dim=-1)

    return {
        "on_off_metrics/on_response_length_mean": _masked_mean(response_length, on_policy_mask),
        "on_off_metrics/off_response_length_mean": _masked_mean(response_length, off_policy_mask),
        "on_off_metrics/on_score": _masked_mean(sequence_score, on_policy_mask),
        "on_off_metrics/off_score": _masked_mean(sequence_score, off_policy_mask),
        "on_off_metrics/off_example_ratio": off_policy_mask.float().mean().item(),
    }


def _compute_rollout_diversity_metrics(batch: DataProto, uids: np.ndarray) -> dict[str, float]:
    """Measure whether the ``n`` responses of a prompt are actually distinct.

    HPT's switch decision depends on best-of-n exploration, so a rollout path
    that accidentally correlates or duplicates per-prompt requests can look
    superficially valid while producing many more all-failed groups.
    """
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    unique_counts = []
    for uid in np.unique(uids):
        signatures = set()
        for row in np.flatnonzero(uids == uid).tolist():
            length = int(response_mask[row].sum().item())
            token_ids = responses[row, :length].detach().cpu().tolist()
            signatures.add(tuple(token_ids))
        unique_counts.append(len(signatures))

    rollout_n = max((int((uids == uid).sum()) for uid in np.unique(uids)), default=0)
    total_responses = len(uids)
    total_unique = sum(unique_counts)
    return {
        "batch/unique_responses_per_group": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "batch/unique_response_ratio": float(total_unique / total_responses) if total_responses else 0.0,
        "batch/duplicate_response_groups": float(sum(count < rollout_n for count in unique_counts)),
    }


def _tokenize_target(tokenizer, target: str, response_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer(target, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    eos_id = tokenizer.eos_token_id
    # The reference off-policy rollout appends EOS unconditionally and then
    # truncates the complete response to the configured response width.
    if eos_id is not None:
        token_ids.append(eos_id)
    token_ids = token_ids[:response_width]
    if not token_ids:
        raise ValueError("HPT produced an empty tokenized target")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        if eos_id is None:
            raise ValueError("HPT requires tokenizer.pad_token_id or tokenizer.eos_token_id")
        pad_id = eos_id
    response = torch.full((response_width,), pad_id, dtype=torch.long)
    response_mask = torch.zeros(response_width, dtype=torch.long)
    response[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    response_mask[: len(token_ids)] = 1
    return response, response_mask


def _replace_with_target(batch: DataProto, row: int, tokenizer) -> None:
    """Replace one rollout in-place with its supervised target (text-only)."""
    if batch.batch["position_ids"].ndim != 2:
        raise NotImplementedError("HPT currently supports text-only rollouts; multimodal position_ids were found")

    target = batch.non_tensor_batch["hpt_target"][row]
    response_width = batch.batch["responses"].shape[-1]
    response, response_mask = _tokenize_target(tokenizer, str(target), response_width)
    device = batch.batch["responses"].device
    response = response.to(device=device, dtype=batch.batch["responses"].dtype)
    response_mask = response_mask.to(device=device, dtype=batch.batch["response_mask"].dtype)

    batch.batch["responses"][row] = response
    batch.batch["response_mask"][row] = response_mask
    prompt_mask = batch.batch["attention_mask"][row, : batch.batch["prompts"].shape[-1]]
    attention_mask = torch.cat([prompt_mask, response_mask.to(prompt_mask.dtype)])
    batch.batch["attention_mask"][row] = attention_mask
    batch.batch["input_ids"][row] = torch.cat([batch.batch["prompts"][row], response])
    batch.batch["position_ids"][row] = compute_position_id_with_mask(attention_mask)


def apply_hpt(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    tokenizer,
    config: DictConfig | dict,
    dp_size: int,
    compute_score=compute_hpt_reward,
) -> tuple[DataProto, torch.Tensor, dict[str, float]]:
    """Apply HPT routing after verification and before old-log-prob inference.

    ``strategy=switch`` implements Algorithm 1 from Unify-Post-Training:
    prompt groups with at most ``switch_gate`` successful rollouts become one
    supervised trajectory; all other groups keep their on-policy rollouts.

    ``strategy=mix`` keeps all on-policy rollouts and replaces the first one in
    each group with a supervised trajectory.  It is provided for reproducing
    the repository's SFT+RL/LUFFY-style ablations.

    ``strategy=luffy_passk0`` keeps every on-policy rollout.  For a prompt
    whose pass@k is zero it additionally appends one offline target with the
    same UID.  The failed online trajectories and successful target therefore
    share one GRPO group, while prompts with pass@k > 0 remain pure GRPO.
    """
    strategy = _get(config, "strategy", "switch")
    supported_strategies = {"switch", "mix", "luffy_passk0"}
    if strategy not in supported_strategies:
        raise ValueError(f"Unsupported HPT strategy {strategy!r}; expected one of {sorted(supported_strategies)}")
    if "hpt_target" not in batch.non_tensor_batch:
        raise KeyError("HPT requires data.custom_cls=HPTDataset so that 'hpt_target' is available")
    if "uid" not in batch.non_tensor_batch:
        raise KeyError("HPT requires the PPO trainer's per-prompt 'uid' field")

    success_reward = float(_get(config, "success_reward", 1.0))
    failure_reward = float(_get(config, "failure_reward", 0.0))
    format_reward = float(_get(config, "format_reward", -1.0))
    switch_gate = int(_get(config, "switch_gate", 0))
    sft_loss_coef = float(_get(config, "sft_loss_coef", 1.0))
    offline_loss_type = _get(config, "offline_loss_type", "sft")
    offline_modes = {"sft": 0, "off_policy": 1, "srft": 2, "dft": 3}
    if offline_loss_type not in offline_modes:
        raise ValueError(
            f"Unsupported HPT offline_loss_type {offline_loss_type!r}; expected one of {sorted(offline_modes)}"
        )
    if strategy == "luffy_passk0" and offline_loss_type != "off_policy":
        raise ValueError("hpt.strategy=luffy_passk0 requires hpt.offline_loss_type=off_policy")
    off_policy_beta = float(_get(config, "off_policy_beta", 0.1))
    if off_policy_beta <= 0:
        raise ValueError("hpt.off_policy_beta must be positive")
    scores = reward_tensor.sum(dim=-1)
    uids = batch.non_tensor_batch["uid"]
    metrics = _compute_batch_metrics(
        scores,
        uids,
        success_reward=success_reward,
        failure_reward=failure_reward,
        format_reward=format_reward,
    )
    metrics.update(_compute_rollout_diversity_metrics(batch, uids))

    solved_groups = 0
    sft_groups = 0
    sorted_uids = np.unique(uids)
    uid_indices: dict[Any, list[int]] = {}
    use_sft_by_uid: dict[Any, bool] = {}
    for uid in sorted_uids:
        indices = np.flatnonzero(uids == uid).tolist()
        uid_indices[uid] = indices
        successes = int(torch.isclose(scores[indices], scores.new_tensor(success_reward)).sum())
        solved_groups += int(successes > 0)
        if strategy == "mix":
            use_sft = True
        elif strategy == "luffy_passk0":
            # pass@k is binary at prompt level: any successful rollout routes
            # to pure GRPO; only an all-failed group receives LUFFY guidance.
            use_sft = successes == 0
        else:
            use_sft = successes <= switch_gate
        use_sft_by_uid[uid] = use_sft
        sft_groups += int(use_sft)

    if strategy == "switch":
        # Reference keeps solved rollout rows in their original global order,
        # then appends one target row for each failed UID in np.unique order.
        solved_rows = [row for row, uid in enumerate(uids) if not use_sft_by_uid[uid]]
        target_rows = [uid_indices[uid][0] for uid in sorted_uids if use_sft_by_uid[uid]]
        keep = solved_rows + target_rows
        sft_flags = [False] * len(solved_rows) + [True] * len(target_rows)
    elif strategy == "luffy_passk0":
        # Retain all k online rollouts, then append exactly one target for each
        # pass@k=0 prompt.  A source-index set cannot represent this because
        # the target is a duplicate of an online row before replacement.
        online_rows = list(range(len(uids)))
        target_rows = [uid_indices[uid][0] for uid in sorted_uids if use_sft_by_uid[uid]]
        keep = online_rows + target_rows
        sft_flags = [False] * len(online_rows) + [True] * len(target_rows)
    else:
        # Preserve the existing noncanonical mix path; Phase 4 only changes switch.
        keep = []
        sft_flags = []
        for uid in dict.fromkeys(uids.tolist()):
            indices = uid_indices[uid]
            keep.extend(indices)
            sft_flags.extend([use_sft_by_uid[uid]] + [False] * (len(indices) - 1))

    selected = batch.select_idxs(keep)
    selected_rewards = reward_tensor[torch.tensor(keep, device=reward_tensor.device)]
    sft_mask = torch.tensor(sft_flags, dtype=torch.bool)
    valid_mask = torch.ones(len(keep), dtype=torch.bool)
    prefix_mask = torch.zeros_like(selected.batch["response_mask"], dtype=torch.bool)

    for row in torch.nonzero(sft_mask, as_tuple=False).flatten().tolist():
        _replace_with_target(selected, row, tokenizer)
        prefix_mask[row] = selected.batch["response_mask"][row].bool()
        valid_response_length = int(selected.batch["response_mask"][row].sum().item())
        valid_response_ids = selected.batch["responses"][row, :valid_response_length]
        decoded_response = decode_legacy_response(tokenizer, valid_response_ids)
        reward_model = selected.non_tensor_batch["reward_model"][row]
        response_width = selected.batch["responses"].shape[-1]
        eos_token_id = tokenizer.eos_token_id
        ends_with_eos = (
            eos_token_id is not None
            and valid_response_length > 0
            and int(valid_response_ids[-1].item()) == eos_token_id
        )
        target_reward = compute_score(
            data_source=selected.non_tensor_batch["data_source"][row],
            solution_str=decoded_response,
            ground_truth=reward_model["ground_truth"],
            extra_info={"response_length": valid_response_length},
            # A full-width target ending in EOS is complete; a full-width
            # target without EOS was truncated by ``_tokenize_target``.
            max_response_length=response_width + int(ends_with_eos),
        )
        selected_rewards[row].zero_()
        selected_rewards[row, valid_response_length - 1] = target_reward

    # Keep a diagnostic-only copy for the on-policy rows before removing the
    # public field. Demonstrations do not have rollout-policy log probabilities,
    # and verl recomputes old_log_probs for the complete transformed batch.
    if "rollout_log_probs" in selected.batch:
        selected.batch["hpt_rollout_log_probs"] = selected.batch["rollout_log_probs"].clone()
    for key in ("rollout_log_probs", "old_log_probs", "entropys"):
        if key in selected.batch:
            selected.batch.pop(key)

    selected.batch["hpt_sft_mask"] = sft_mask.to(selected.batch.device)
    selected.batch["prefix_mask"] = prefix_mask.to(selected.batch.device)
    selected.batch["hpt_sft_weight"] = torch.full(
        (len(selected),), sft_loss_coef, dtype=torch.float32, device=selected.batch.device
    )
    selected.batch["hpt_valid_mask"] = valid_mask.to(selected.batch.device)
    selected.batch["hpt_offline_mode"] = torch.full(
        (len(selected),), offline_modes[offline_loss_type], dtype=torch.int8, device=selected.batch.device
    )
    selected.batch["hpt_off_policy_beta"] = torch.full(
        (len(selected),), off_policy_beta, dtype=torch.float32, device=selected.batch.device
    )
    selected.batch["hpt_correct_mask"] = torch.isclose(
        selected_rewards.sum(dim=-1), selected_rewards.new_tensor(success_reward)
    ).to(selected.batch.device)

    # Padding belongs to two later trainer boundaries in the 0.1 dataflow:
    # temporary old-log-prob padding and final post-advantage actor padding.
    del dp_size
    pad_size = 0

    metrics.update(_compute_on_off_metrics(selected, selected_rewards, prefix_mask))
    metrics.update(
        {
            "hpt/sft_groups": float(sft_groups),
            "hpt/rl_groups": float(len(set(uids.tolist())) - sft_groups),
            "hpt/solved_groups": float(solved_groups),
            "hpt/passk_positive_groups": float(solved_groups),
            "hpt/passk_zero_groups": float(len(set(uids.tolist())) - solved_groups),
            "hpt/sft_sequences": float(sft_mask.sum().item()),
            "hpt/online_sequences": float((~sft_mask).sum().item()),
            "hpt/valid_sequences": float(len(keep)),
            "hpt/padded_sequences": float(pad_size),
            "hpt/offline_mode": float(offline_modes[offline_loss_type]),
        }
    )
    return selected, selected_rewards, metrics
