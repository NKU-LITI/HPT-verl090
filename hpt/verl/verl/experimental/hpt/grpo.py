# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Legacy HPT-repository GRPO policy objective on the verl 0.9 runtime."""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig


@register_policy_loss("grpo_compat")
def compute_policy_loss_grpo_compat(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reproduce the policy-loss switches used by the original HPT experiments.

    ``loss_remove_clip`` selects the raw importance-ratio objective. When
    ``loss_remove_token_mean`` is enabled, the masked token loss is summed and
    divided by the fixed response horizon, exactly matching the old
    ``sum() / eos_mask.shape[-1]`` normalization while retaining correct DP
    scaling in the 0.9 runtime.
    """
    if config is None:
        raise ValueError("grpo_compat policy loss requires actor config")

    policy_config = config.policy_loss
    remove_clip = policy_config.get("loss_remove_clip", False)
    remove_token_mean = policy_config.get("loss_remove_token_mean", False)

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    pg_losses_unclipped = -advantages * ratio
    pg_losses_clipped = -advantages * torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    )
    pg_losses = pg_losses_unclipped if remove_clip else torch.maximum(pg_losses_unclipped, pg_losses_clipped)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    if remove_token_mean:
        # agg_loss(token-sum) supplies the DP scaling needed so the averaged
        # FSDP gradient is the global masked sum. The old denominator was the
        # padded response width, not the valid-token count or batch size.
        pg_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=response_mask,
            loss_agg_mode="token-sum",
            **config.global_batch_info,
        ) / response_mask.shape[-1]
    else:
        pg_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    pg_clipfrac = masked_mean(torch.gt(pg_losses_clipped, pg_losses_unclipped).float(), response_mask)
    ppo_kl = masked_mean(-log_ratio, response_mask)
    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
        "grpo_compat/remove_clip": float(remove_clip),
        "grpo_compat/remove_token_mean": float(remove_token_mean),
    }
    return pg_loss, metrics
