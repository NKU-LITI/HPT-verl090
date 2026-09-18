# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Canonical verl 0.1 HPT actor-update compatibility helpers."""

import torch


def legacy_gradient_accumulation(ppo_mini_batch_size: int, ppo_micro_batch_size: int) -> int:
    """Return the fixed 0.1 loss denominator, independent of dynamic partition count."""
    if ppo_micro_batch_size <= 0 or ppo_mini_batch_size % ppo_micro_batch_size != 0:
        raise ValueError(
            "legacy HPT requires ppo_mini_batch_size to be divisible by "
            f"ppo_micro_batch_size, got {ppo_mini_batch_size} and {ppo_micro_batch_size}"
        )
    return ppo_mini_batch_size // ppo_micro_batch_size


def scale_legacy_backward_loss(loss: torch.Tensor, gradient_accumulation: int | None) -> torch.Tensor:
    """Scale each dynamic micro-batch loss by the fixed legacy denominator."""
    if gradient_accumulation is None:
        return loss
    if gradient_accumulation <= 0:
        raise ValueError(f"gradient_accumulation must be positive, got {gradient_accumulation}")
    return loss / gradient_accumulation


def legacy_grad_norm_skip(grad_norm: torch.Tensor, max_grad_norm: float | None) -> tuple[bool, bool]:
    """Return ``(skip, nonfinite)`` for the post-clip, pre-step legacy decision."""
    nonfinite = not bool(torch.isfinite(grad_norm).item())
    too_large = bool(max_grad_norm) and bool((grad_norm > max_grad_norm).item())
    return nonfinite or too_large, nonfinite
