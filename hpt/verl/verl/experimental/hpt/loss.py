# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registered legacy HPT objective for the verl 0.9 runtime."""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl.mix_src.legacy_policy_loss import compute_policy_loss as compute_v01_policy_loss
from verl.mix_src.legacy_policy_loss import compute_sft_pure_loss as compute_v01_sft_pure_loss
from verl.trainer.ppo.core_algos import register_policy_loss
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    if not mask.any():
        return values.sum() * 0.0
    return (values * mask).sum() / mask.sum()


def compute_hpt_legacy_entropy_loss(entropy: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Match the 0.1 HPT actor's current-microbatch valid-token entropy mean."""
    return _masked_mean_or_zero(entropy, response_mask)


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
    """Reproduce the old HPT actor objective for one fixed micro-batch.

    Modes 0 (SFT + on-policy), 1 (combined on/off-policy), and 3 (DFT +
    on-policy) follow the original HPT formulas. In particular, their
    canonical configuration disables ratio clipping and divides the masked RL
    token sum by the padded response width. No verl 0.9 global-batch
    aggregation is used here; the old actor normalized each micro-batch before
    its separate backward call.
    """
    del loss_agg_mode
    if hpt_sft_mask is None:
        raise KeyError("policy_loss.loss_mode=hpt requires 'hpt_sft_mask' in the training batch")
    if config is None:
        raise ValueError("HPT policy loss requires actor config")

    target_rows = hpt_sft_mask.bool()
    valid_rows = torch.ones_like(target_rows) if hpt_valid_mask is None else hpt_valid_mask.bool()
    on_rows = (~target_rows) & valid_rows
    target_rows = target_rows & valid_rows
    effective_mask = response_mask.bool() & valid_rows.unsqueeze(-1)

    mode = 0
    if hpt_offline_mode is not None and valid_rows.any():
        modes = torch.unique(hpt_offline_mode[valid_rows])
        if modes.numel() != 1:
            raise ValueError(f"HPT batch contains multiple offline loss modes: {modes.tolist()}")
        mode = int(modes.item())
    if mode not in {0, 1, 2, 3}:
        raise ValueError(f"Unknown HPT offline loss mode id: {mode}")

    zero = log_prob.sum() * 0.0

    # SRFT is not part of the canonical HPT/LUFFY objective repaired here.
    # Preserve its existing 0.9 implementation while modes 0/1 take the
    # explicit legacy path below.
    if mode == 2:
        sft_loss = zero
        off_policy_loss = zero
        srft_on_loss = zero
        if target_rows.any():
            if entropy is None:
                raise ValueError("HPT SRFT mode requires actor.calculate_entropy=True")
            token_weights = 1.0 if hpt_sft_weight is None else hpt_sft_weight[target_rows, None]
            entropy_mean = masked_mean(entropy, effective_mask, axis=-1).detach()
            token_weights = token_weights * (0.5 * torch.exp(-entropy_mean[target_rows])).unsqueeze(-1)
            sft_loss = _masked_mean_or_zero(-log_prob[target_rows] * token_weights, effective_mask[target_rows])

            beta = 0.1 if hpt_off_policy_beta is None else hpt_off_policy_beta[target_rows, None]
            probability = torch.exp(log_prob[target_rows])
            reshaped_probability = probability / (probability + beta)
            off_policy_loss = _masked_mean_or_zero(
                -advantages[target_rows] * reshaped_probability,
                effective_mask[target_rows],
            )

        if on_rows.any():
            if entropy is None or hpt_correct_mask is None:
                raise ValueError("HPT SRFT mode requires entropy and hpt_correct_mask")
            entropy_mean = masked_mean(entropy, effective_mask, axis=-1).detach()
            positive_advantage = 0.1 * torch.exp(entropy_mean[on_rows])
            correct = hpt_correct_mask[on_rows].bool()
            srft_advantage = torch.where(correct, positive_advantage, -torch.ones_like(positive_advantage))
            srft_on_loss = _masked_mean_or_zero(
                -srft_advantage.unsqueeze(-1) * log_prob[on_rows],
                effective_mask[on_rows],
            )

        loss = sft_loss + off_policy_loss + srft_on_loss
        metrics = {
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.0,
            "actor/pg_clipfrac_lower": 0.0,
            "hpt/rl_loss": 0.0,
            "hpt/sft_loss": sft_loss.detach().item(),
            "hpt/off_policy_loss": off_policy_loss.detach().item(),
            "hpt/srft_on_loss": srft_on_loss.detach().item(),
            "hpt/rl_fraction": on_rows.float().mean().detach().item(),
            "hpt/sft_fraction": target_rows.float().mean().detach().item(),
        }
        return loss, metrics

    policy_config = config.get("policy_loss", {})
    remove_clip = bool(policy_config.get("loss_remove_clip", False))
    remove_token_mean = bool(policy_config.get("loss_remove_token_mean", False))

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    on_losses_unclipped = -advantages * ratio
    on_losses_clipped = -advantages * torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    )
    on_token_loss = on_losses_unclipped if remove_clip else torch.maximum(on_losses_unclipped, on_losses_clipped)
    if rollout_is_weights is not None:
        on_token_loss = on_token_loss * rollout_is_weights

    response_width = effective_mask.shape[-1]
    target_token_mask = effective_mask & target_rows.unsqueeze(-1)
    on_token_mask = effective_mask & on_rows.unsqueeze(-1)

    sft_loss = zero
    off_policy_loss = zero
    used_v01_policy_source = False
    if mode in {0, 3}:
        # The canonical HPT path executes the old function body directly.  The
        # manual 0.9 path remains only for optional rollout importance weights,
        # which did not exist in verl 0.1.
        if rollout_is_weights is None and on_rows.any():
            rl_loss, pg_clipfrac, ppo_kl = compute_v01_policy_loss(
                old_log_prob=old_log_prob[on_rows],
                log_prob=log_prob[on_rows],
                advantages=advantages[on_rows],
                eos_mask=effective_mask[on_rows],
                cliprange=config.clip_ratio,
                loss_remove_token_mean=remove_token_mean,
                loss_remove_clip=remove_clip,
            )
            used_v01_policy_source = True
        elif remove_token_mean:
            rl_loss = (on_token_loss * on_token_mask).sum() / response_width
        else:
            rl_loss = _masked_mean_or_zero(on_token_loss, on_token_mask)

        if target_rows.any() and mode == 0:
            target_log_prob = log_prob[target_rows]
            if hpt_sft_weight is not None:
                target_log_prob = target_log_prob * hpt_sft_weight[target_rows, None]
            sft_loss = compute_v01_sft_pure_loss(target_log_prob, effective_mask[target_rows])

        if target_rows.any() and mode == 3:
            # Legacy DFT objective from the verl 0.7 HPT implementation:
            # -stopgrad(pi_theta) * log(pi_theta).
            token_weights = torch.exp(log_prob).detach()
            sft_loss = _masked_mean_or_zero(-token_weights * log_prob, target_token_mask)

        loss = rl_loss + sft_loss
    else:
        beta = 0.1 if hpt_off_policy_beta is None else hpt_off_policy_beta[:, None]
        probability = torch.exp(log_prob)
        off_token_loss = -advantages * probability / (probability + beta)

        if remove_token_mean:
            rl_loss = (on_token_loss * on_token_mask).sum() / response_width
            off_policy_loss = (off_token_loss * target_token_mask).sum() / response_width
        else:
            denominator = effective_mask.sum()
            rl_loss = (on_token_loss * on_token_mask).sum() / denominator
            off_policy_loss = (off_token_loss * target_token_mask).sum() / denominator

        # These contributions use one shared denominator, exactly matching
        # the old combined-token matrix before its single normalization.
        loss = rl_loss + off_policy_loss

    if not used_v01_policy_source:
        clip_mask = on_token_mask if mode in {0, 3} else effective_mask
        pg_clipfrac = _masked_mean_or_zero(
            torch.gt(on_losses_clipped, on_losses_unclipped).float(),
            clip_mask,
        )
        ppo_kl_mask = on_token_mask if mode in {0, 3} else effective_mask
        ppo_kl = _masked_mean_or_zero(-log_ratio, ppo_kl_mask)
    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
        "hpt/rl_loss": rl_loss.detach().item(),
        "hpt/sft_loss": sft_loss.detach().item(),
        "hpt/off_policy_loss": off_policy_loss.detach().item(),
        "hpt/srft_on_loss": 0.0,
        "hpt/rl_fraction": on_rows.float().mean().detach().item(),
        "hpt/sft_fraction": target_rows.float().mean().detach().item(),
        "hpt/remove_clip": float(remove_clip),
        "hpt/remove_token_mean": float(remove_token_mean),
    }
    return loss, metrics
