# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Executable verl-0.1 HPT actor-update contract.

The source statements represented here come from
``verl/mix_src/mix_fsdp_worker.py::MIXActorRolloutRefWorker.__init__`` and
``mix_actor.py::MIXDataParallelPPOActor.update_policy/_optimizer_step``.  The
worker first converts configured prompt-level sizes to per-rank sequence
sizes; the actor then splits its already DP-sharded batch with those values.
"""

import torch


def legacy_constant_schedule_scale(current_step: int, num_warmup_steps: int) -> float:
    """Direct 0.1 constant-schedule lambda, including its zero-warmup quirk."""
    return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))


def normalize_actor_batch_sizes(
    ppo_mini_batch_size: int,
    ppo_micro_batch_size: int,
    rollout_n: int,
    world_size: int,
    sequence_parallel_size: int = 1,
) -> tuple[int, int]:
    """Direct 0.1 worker-side prompt-to-local-sequence normalization."""
    dp_size = world_size // sequence_parallel_size
    local_mini_batch_size = ppo_mini_batch_size * rollout_n // dp_size
    local_micro_batch_size = ppo_micro_batch_size * rollout_n // dp_size
    return local_mini_batch_size, local_micro_batch_size


def split_local_mini_batches(batch, ppo_mini_batch_size: int):
    """Direct 0.1 statement: ``batch.split(self.config.ppo_mini_batch_size)``."""
    return batch.split(ppo_mini_batch_size)


def legacy_gradient_accumulation(ppo_mini_batch_size: int, ppo_micro_batch_size: int) -> int:
    """Direct 0.1 fixed loss denominator for every dynamic micro-batch."""
    if ppo_micro_batch_size <= 0 or ppo_mini_batch_size % ppo_micro_batch_size != 0:
        raise ValueError(
            "legacy HPT requires ppo_mini_batch_size to be divisible by "
            f"ppo_micro_batch_size, got {ppo_mini_batch_size} and {ppo_micro_batch_size}"
        )
    return ppo_mini_batch_size // ppo_micro_batch_size


def scale_legacy_backward_loss(loss: torch.Tensor, gradient_accumulation: int | None) -> torch.Tensor:
    """Direct 0.1 statement: ``loss = policy_loss / self.gradient_accumulation``."""
    if gradient_accumulation is None:
        return loss
    if gradient_accumulation <= 0:
        raise ValueError(f"gradient_accumulation must be positive, got {gradient_accumulation}")
    return loss / gradient_accumulation


def legacy_grad_norm_skip(grad_norm: torch.Tensor, max_grad_norm: float | None) -> tuple[bool, bool]:
    """Preserve the 0.1 post-clip whole-optimizer-step skip decision."""
    nonfinite = not bool(torch.isfinite(grad_norm).item())
    too_large = bool(max_grad_norm) and bool((grad_norm > max_grad_norm).item())
    return nonfinite or too_large, nonfinite
