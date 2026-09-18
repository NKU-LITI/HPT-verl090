# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Canonical verl 0.1 HPT batch-layout compatibility helpers."""

import torch

from verl import DataProto
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

HPT_LEGACY_PADDING_DIVISOR = 8


def pad_hpt_batch_like_reference(
    batch: DataProto,
    divisor: int,
    *,
    mark_actor_padding: bool,
) -> tuple[DataProto, int]:
    """Repeat the final row to a divisor, matching the two legacy padding sites."""
    if divisor <= 0:
        raise ValueError(f"padding divisor must be positive, got {divisor}")
    if len(batch) == 0:
        raise ValueError("cannot pad an empty HPT batch")

    if mark_actor_padding:
        device = batch.batch.device
        batch.batch["whether_pad"] = torch.zeros(len(batch), dtype=torch.bool, device=device)
        batch.batch["hpt_valid_mask"] = torch.ones(len(batch), dtype=torch.bool, device=device)

    pad_size = (-len(batch)) % divisor
    if pad_size == 0:
        return batch, 0

    last_row = batch.select_idxs([len(batch) - 1]).repeat(pad_size)
    padded = DataProto.concat([batch, last_row])
    if mark_actor_padding:
        padded.batch["whether_pad"][-pad_size:] = True
        padded.batch["hpt_valid_mask"][-pad_size:] = False
    return padded, pad_size


def balance_hpt_batch_like_reference(batch: DataProto, dp_size: int) -> tuple[list[list[int]], list[int]]:
    """Apply the 0.1 row-level equal-size Karmarkar–Karp DP balance."""
    attention_mask = batch.batch["attention_mask"]
    sequence_costs = attention_mask.reshape(len(batch), -1).sum(dim=-1).tolist()
    partitions = get_seqlen_balanced_partitions(sequence_costs, k_partitions=dp_size, equal_size=True)
    global_indices = torch.tensor(
        [row for partition in partitions for row in partition],
        dtype=torch.long,
        device=attention_mask.device,
    )
    batch.reorder(global_indices)
    return partitions, sequence_costs
