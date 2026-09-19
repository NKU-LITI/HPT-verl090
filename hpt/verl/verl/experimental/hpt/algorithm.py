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
from verl.experimental.hpt.reward_manager import decode_legacy_response
from verl.utils.model import compute_position_id_with_mask


def _get(config: DictConfig | dict, key: str, default: Any) -> Any:
    return config.get(key, default)


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
) -> tuple[DataProto, torch.Tensor, dict[str, float]]:
    """Apply HPT routing after verification and before old-log-prob inference.

    ``strategy=switch`` implements Algorithm 1 from Unify-Post-Training:
    prompt groups with at most ``switch_gate`` successful rollouts become one
    supervised trajectory; all other groups keep their on-policy rollouts.

    ``strategy=mix`` keeps all on-policy rollouts and replaces the first one in
    each group with a supervised trajectory.  It is provided for reproducing
    the repository's SFT+RL/LUFFY-style ablations.
    """
    strategy = _get(config, "strategy", "switch")
    if strategy not in {"switch", "mix"}:
        raise ValueError(f"Unsupported HPT strategy {strategy!r}; expected 'switch' or 'mix'")
    if "hpt_target" not in batch.non_tensor_batch:
        raise KeyError("HPT requires data.custom_cls=HPTDataset so that 'hpt_target' is available")
    if "uid" not in batch.non_tensor_batch:
        raise KeyError("HPT requires the PPO trainer's per-prompt 'uid' field")

    success_reward = float(_get(config, "success_reward", 1.0))
    switch_gate = int(_get(config, "switch_gate", 0))
    sft_loss_coef = float(_get(config, "sft_loss_coef", 1.0))
    offline_loss_type = _get(config, "offline_loss_type", "sft")
    offline_modes = {"sft": 0, "off_policy": 1, "srft": 2, "dft": 3}
    if offline_loss_type not in offline_modes:
        raise ValueError(
            f"Unsupported HPT offline_loss_type {offline_loss_type!r}; "
            f"expected one of {sorted(offline_modes)}"
        )
    off_policy_beta = float(_get(config, "off_policy_beta", 0.1))
    if off_policy_beta <= 0:
        raise ValueError("hpt.off_policy_beta must be positive")
    scores = reward_tensor.sum(dim=-1)
    uids = batch.non_tensor_batch["uid"]

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
        use_sft = strategy == "mix" or successes <= switch_gate
        use_sft_by_uid[uid] = use_sft
        sft_groups += int(use_sft)

    if strategy == "switch":
        # Reference keeps solved rollout rows in their original global order,
        # then appends one target row for each failed UID in np.unique order.
        solved_rows = [row for row, uid in enumerate(uids) if not use_sft_by_uid[uid]]
        target_rows = [uid_indices[uid][0] for uid in sorted_uids if use_sft_by_uid[uid]]
        keep = solved_rows + target_rows
        sft_source_rows = set(target_rows)
    else:
        # Preserve the existing noncanonical mix path; Phase 4 only changes switch.
        keep = []
        sft_source_rows = set()
        for uid in dict.fromkeys(uids.tolist()):
            indices = uid_indices[uid]
            keep.extend(indices)
            if use_sft_by_uid[uid]:
                sft_source_rows.add(indices[0])

    selected = batch.select_idxs(keep)
    selected_rewards = reward_tensor[torch.tensor(keep, device=reward_tensor.device)]
    sft_mask = torch.tensor([source in sft_source_rows for source in keep], dtype=torch.bool)
    valid_mask = torch.ones(len(keep), dtype=torch.bool)
    prefix_mask = torch.zeros_like(selected.batch["response_mask"], dtype=torch.bool)

    for row in torch.nonzero(sft_mask, as_tuple=False).flatten().tolist():
        _replace_with_target(selected, row, tokenizer)
        prefix_mask[row] = selected.batch["response_mask"][row].bool()
        valid_response_length = int(selected.batch["response_mask"][row].sum().item())
        valid_response_ids = selected.batch["responses"][row, :valid_response_length]
        decoded_response = decode_legacy_response(tokenizer, valid_response_ids)
        reward_model = selected.non_tensor_batch["reward_model"][row]
        target_reward = compute_hpt_reward(
            data_source=selected.non_tensor_batch["data_source"][row],
            solution_str=decoded_response,
            ground_truth=reward_model["ground_truth"],
        )
        selected_rewards[row].zero_()
        selected_rewards[row, valid_response_length - 1] = target_reward

    # Demonstrations do not have rollout-policy log probabilities.  verl will
    # recompute old_log_probs for the complete transformed batch below.
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

    metrics = {
        "hpt/sft_groups": float(sft_groups),
        "hpt/rl_groups": float(len(set(uids.tolist())) - sft_groups),
        "hpt/solved_groups": float(solved_groups),
        "hpt/sft_sequences": float(sft_mask.sum().item()),
        "hpt/valid_sequences": float(len(keep)),
        "hpt/padded_sequences": float(pad_size),
        "hpt/offline_mode": float(offline_modes[offline_loss_type]),
    }
    return selected, selected_rewards, metrics
