# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Direct verl-0.1 HPT policy/SFT formulas.

Source bodies:

* ``verl/trainer/ppo/core_algos.py::compute_policy_loss``
* ``verl/mix_src/mix_core_alg.py::compute_sft_pure_loss``

Only the old debug ``print`` was removed.  Keeping the old masked-mean formula
here is intentional: verl 0.9 adds an epsilon to its shared implementation.
"""

import torch


def _masked_mean(values, mask, axis=None):
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = _masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    if loss_remove_clip is False:
        pg_losses = torch.max(pg_losses, pg_losses2)

    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
    else:
        pg_loss = _masked_mean(pg_losses, eos_mask)

    pg_clipfrac = _masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    return _masked_mean(sft_losses, eos_mask)

