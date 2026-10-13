# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Naive reward manager that exposes the valid response token count to compute_score."""

from __future__ import annotations

import numpy as np
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

RESPONSE_LENGTH_KEY = "response_length"  # Must match reward.RESPONSE_LENGTH_KEY.


class LengthAwareNaiveRewardManager(NaiveRewardManager):
    """Adds ``extra_info["response_length"]`` (valid response tokens) before scoring.

    The decoded ``solution_str`` cannot recover the token count, so the reward
    function needs it here to reject responses cut off at ``max_response_length``.
    """

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # Same item NaiveRewardManager scores; slicing gives a fresh dict.
        data_item = data[0]
        response_length = data_item.batch["responses"].shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        # Copy: extra_info dicts are shared between the n rollouts of one prompt.
        extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})
        extra_info[RESPONSE_LENGTH_KEY] = valid_response_length
        data.non_tensor_batch["extra_info"] = np.array([extra_info], dtype=object)
        return await super().run_single(data)
