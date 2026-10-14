# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Canonical verl 0.1 HPT reward compatibility wrapper."""

from __future__ import annotations

if __package__:
    from .legacy_entropy_math import compute_score as legacy_entropy_math_score
else:  # Support the legacy file-path loader as well as pkg:// imports.
    from verl.experimental.hpt.legacy_entropy_math import compute_score as legacy_entropy_math_score


def legacy_reward_input(solution_str: str) -> str:
    """Reproduce ``RewardManager``'s reward_impl_version=6 text prefix."""
    return "<think>\n" + solution_str


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Run the exact entropy_math verifier selected by canonical HPT."""
    del data_source, extra_info, kwargs
    return float(bool(legacy_entropy_math_score(legacy_reward_input(solution_str), ground_truth)))


def compute_score_in_process(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Pickle-stable entry point used by the legacy reward process pool."""
    return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
