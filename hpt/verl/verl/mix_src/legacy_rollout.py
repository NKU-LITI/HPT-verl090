# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Executable verl-0.7 HPT rollout-scheduling contract.

The 0.7 agent-loop manager split the repeated rollout batch across eight
workers and submitted every request without a priority field.  Consequently
all requests reached vLLM with its default priority (zero).  verl 0.9 assigns
a distinct increasing priority to every row.  That is not merely an execution
optimization for HPT: routing depends on the joint distribution of the eight
responses belonging to one prompt.
"""

import numpy as np


def legacy_equal_request_priorities(batch_size: int) -> np.ndarray:
    """Return the effective request priorities of the 0.7 HPT agent loop."""
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    return np.zeros(batch_size, dtype=np.int64)
