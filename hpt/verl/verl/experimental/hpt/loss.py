# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Registered mixed PPO/SFT objective used by HPT."""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, register_policy_loss
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig


@register_policy_loss("hpt")
def compute_policy_loss_hpt(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    hpt_sft_mask: torch.Tensor | None = None,
    hpt_valid_mask: torch.Tensor | None = None,
    hpt_sft_weight: torch.Tensor | None = None,
    hpt_offline_mode: torch.Tensor | None = None,
    hpt_off_policy_beta: torch.Tensor | None = None,
    hpt_correct_mask: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine PPO on rollout rows with NLL on demonstration rows."""
    if hpt_sft_mask is None:
        raise KeyError("policy_loss.loss_mode=hpt requires 'hpt_sft_mask' in the training batch")
    if config is None:
        raise ValueError("HPT policy loss requires actor config")

    sft_rows = hpt_sft_mask.bool()
    valid_rows = torch.ones_like(sft_rows) if hpt_valid_mask is None else hpt_valid_mask.bool()
    rl_rows = (~sft_rows) & valid_rows
    sft_rows = sft_rows & valid_rows
    mode = 0
    if hpt_offline_mode is not None and valid_rows.any():
        modes = torch.unique(hpt_offline_mode[valid_rows])
        if modes.numel() != 1:
            raise ValueError(f"HPT batch contains multiple offline loss modes: {modes.tolist()}")
        mode = int(modes.item())
    if mode not in {0, 1, 2}:
        raise ValueError(f"Unknown HPT offline loss mode id: {mode}")

    zero = log_prob.sum() * 0.0
    if rl_rows.any() and mode != 2:
        rl_loss, rl_metrics = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob[rl_rows],
            log_prob=log_prob[rl_rows],
            advantages=advantages[rl_rows],
            response_mask=response_mask[rl_rows],
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights[rl_rows] if rollout_is_weights is not None else None,
        )
    else:
        rl_loss = zero
        rl_metrics = {
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.0,
            "actor/pg_clipfrac_lower": 0.0,
        }

    sft_loss = zero
    off_policy_loss = zero
    srft_on_loss = zero
    if sft_rows.any():
        if mode in {0, 2}:
            token_weights = 1.0 if hpt_sft_weight is None else hpt_sft_weight[sft_rows, None]
            if mode == 2:
                if entropy is None:
                    raise ValueError("HPT SRFT mode requires actor.calculate_entropy=True")
                entropy_mean = masked_mean(entropy, response_mask, axis=-1).detach()
                token_weights = token_weights * (0.5 * torch.exp(-entropy_mean[sft_rows])).unsqueeze(-1)
            sft_loss = masked_mean(-log_prob[sft_rows] * token_weights, response_mask[sft_rows])

        if mode in {1, 2}:
            beta = 0.1 if hpt_off_policy_beta is None else hpt_off_policy_beta[sft_rows, None]
            probability = torch.exp(log_prob[sft_rows])
            reshaped_probability = probability / (probability + beta)
            off_policy_loss = masked_mean(
                -advantages[sft_rows] * reshaped_probability,
                response_mask[sft_rows],
            )

    if mode == 2 and rl_rows.any():
        if entropy is None or hpt_correct_mask is None:
            raise ValueError("HPT SRFT mode requires entropy and hpt_correct_mask")
        entropy_mean = masked_mean(entropy, response_mask, axis=-1).detach()
        positive_advantage = 0.1 * torch.exp(entropy_mean[rl_rows])
        correct = hpt_correct_mask[rl_rows].bool()
        srft_advantage = torch.where(correct, positive_advantage, -torch.ones_like(positive_advantage))
        srft_on_loss = masked_mean(
            -srft_advantage.unsqueeze(-1) * log_prob[rl_rows],
            response_mask[rl_rows],
        )

    loss = rl_loss + sft_loss + off_policy_loss + srft_on_loss
    metrics = dict(rl_metrics)
    metrics.update(
        {
            "hpt/rl_loss": rl_loss.detach().item(),
            "hpt/sft_loss": sft_loss.detach().item(),
            "hpt/off_policy_loss": off_policy_loss.detach().item(),
            "hpt/srft_on_loss": srft_on_loss.detach().item(),
            "hpt/rl_fraction": rl_rows.float().mean().detach().item(),
            "hpt/sft_fraction": sft_rows.float().mean().detach().item(),
        }
    )
    return loss, metrics
