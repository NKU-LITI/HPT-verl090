# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Binary math verifier used by the migrated HPT recipe."""

from __future__ import annotations

from verl.utils.reward_score.math_verify import compute_score as math_verify_score
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Return the binary 0/1 verifier signal expected by HPT's switch gate."""
    del data_source, extra_info, kwargs
    references = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth]
    result = 0.0
    for reference in references:
        reference = str(reference)
        boxed = last_boxed_only_string(reference)
        if boxed is not None:
            reference = remove_boxed(boxed)
        result = max(result, math_verify_score(model_output=solution_str, ground_truth=reference))
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        result = result["score"]
    return float(bool(result))
