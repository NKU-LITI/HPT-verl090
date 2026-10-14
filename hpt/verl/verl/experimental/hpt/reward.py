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
    from .answer_rules import choice_letters, parse_choice, split_label, split_top_level, strip_unit
    from .legacy_entropy_math import compute_score as legacy_entropy_math_score
    from .legacy_entropy_math import extract_answer
else:  # Support the legacy file-path loader as well as pkg:// imports.
    from verl.experimental.hpt.answer_rules import (
        choice_letters,
        parse_choice,
        split_label,
        split_top_level,
        strip_unit,
    )
    from verl.experimental.hpt.legacy_entropy_math import compute_score as legacy_entropy_math_score
    from verl.experimental.hpt.legacy_entropy_math import extract_answer

THINK_END = "</think>"
# Kept as a public compatibility key for reward managers and rollout tools.
# The current scorer deliberately does not reject a response by token length.
RESPONSE_LENGTH_KEY = "response_length"


def legacy_reward_input(solution_str: str) -> str:
    """Reproduce ``RewardManager``'s reward_impl_version=6 text prefix."""
    return "<think>\n" + solution_str


def final_response(solution_str: str) -> str | None:
    """Return the text after the last ``</think>``, or None if it is missing.

    The verifier then takes the last ``\\boxed{}`` in this text as the answer.
    """
    if THINK_END not in solution_str:
        return None
    return solution_str.rsplit(THINK_END, 1)[1]


def ground_truth_answers(ground_truth) -> list[str]:
    """Normalize a ground truth (str, number, or list of them) to unboxed strings."""
    answers = []
    for gt in ground_truth if isinstance(ground_truth, list) else [ground_truth]:
        if isinstance(gt, (int, float)):
            gt = str(gt)
        if not isinstance(gt, str):
            continue
        if "\\boxed" in gt:
            gt = extract_answer(gt)
            if gt is None:
                continue
        answers.append(gt)
    return answers


GT_SEPARATORS = ","
PRED_SEPARATORS = ",，;；"


def field_equal(pred: str, gt: str) -> bool:
    """Compare one answer field; labels must match when both sides have one.

    Units are stripped from both sides first, so 30° matches 30 but 25% does
    not match 0.25 (legacy alone would convert the percent).
    """
    pred_label, pred = split_label(pred)
    gt_label, gt = split_label(gt)
    if pred_label is not None and gt_label is not None and pred_label != gt_label:
        return False
    pred, gt = strip_unit(pred), strip_unit(gt)
    # [L2] The legacy verifier extracts the last box, so pass the field boxed.
    return bool(legacy_entropy_math_score("\\boxed{" + pred + "}", gt))


def grade(final: str, pred: str, gt: str) -> bool:
    """Grade the extracted prediction against one ground-truth answer."""
    letters = choice_letters(gt)
    if letters is not None:
        # [L1] Legacy verdict first; parse the label format only if it fails.
        if legacy_entropy_math_score(final, gt):
            return True
        return parse_choice(pred) == letters
    gt_fields = split_top_level(gt, GT_SEPARATORS)
    if gt_fields is not None and len(gt_fields) > 1:
        # Ordered blanks: never grade the whole string, legacy may ignore order.
        pred_fields = split_top_level(pred, PRED_SEPARATORS)
        if pred_fields is None or len(pred_fields) != len(gt_fields):
            return False
        return all(field_equal(p, g) for p, g in zip(pred_fields, gt_fields))
    return field_equal(pred, gt)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    """Grade the last box after ``</think>``."""
    del data_source, extra_info, kwargs
    final = final_response(solution_str)
    if final is None:
        return 0.0
    pred = extract_answer(final)
    if pred is None:
        return 0.0
    return float(any(grade(final, pred, gt) for gt in ground_truth_answers(ground_truth)))


def compute_score_in_process(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Pickle-stable entry point used by the legacy reward process pool."""
    return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
