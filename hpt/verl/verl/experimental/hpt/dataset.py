# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dataset adapter that preserves the supervised trajectory required by HPT."""

from __future__ import annotations

from typing import Any

from torch.utils.data import SequentialSampler

from verl.mix_src.legacy_sampler import HPTLegacyRandomSampler
from verl.utils.dataset.rl_dataset import RLHFDataset


def create_hpt_train_sampler(data_config, dataset):
    """Create a train sampler with the canonical HPT 0.1 ordering semantics."""
    if data_config.get("shuffle", True):
        seed = data_config.get("seed", 47)
        return HPTLegacyRandomSampler(dataset, seed=47 if seed is None else seed)
    return SequentialSampler(dataset)


def _target_to_text(target: Any) -> str:
    """Normalize the target formats used by the original HPT datasets."""
    if target is None:
        return ""
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        if "content" not in target:
            raise ValueError("HPT target dict must contain a 'content' field")
        return str(target["content"])
    if isinstance(target, (list, tuple)):
        if not target:
            return ""
        # Original Unify-Post-Training parquet files store one assistant
        # message as: target=[{"role": "assistant", "content": "..."}].
        return _target_to_text(target[0])
    if hasattr(target, "tolist"):
        return _target_to_text(target.tolist())
    raise TypeError(f"Unsupported HPT target type: {type(target)!r}")


def _strip_duplicate_think_prefix(prompt_text: str, target_text: str) -> str:
    """Match the 0.1 dataset's prompt/target boundary normalization."""
    think_prefix = "<think>\n"
    if prompt_text.endswith(think_prefix) and target_text.startswith(think_prefix):
        return target_text[len(think_prefix) :]
    return target_text


class HPTDataset(RLHFDataset):
    """verl 0.9 RLHF dataset with a normalized ``hpt_target`` column.

    Configuration keys added by this adapter:

    - ``target_key``: source parquet column, default ``target``.
    - ``require_target``: fail on missing/empty targets while loading.  It is
      disabled by default because verl shares one data config between train and
      target-less validation datasets; the HPT training path still fails fast
      if it actually needs an empty target.
    """

    def __getitem__(self, item):
        row = super().__getitem__(item)
        target_key = self.config.get("target_key", "target")
        target = row.pop(target_key, None)
        target_text = _target_to_text(target)
        apply_kwargs = dict(self.apply_chat_template_kwargs)
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)
        prompt_text = self.tokenizer.apply_chat_template(
            row["raw_prompt"],
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        target_text = _strip_duplicate_think_prefix(prompt_text, target_text)
        if self.config.get("require_target", False) and not target_text.strip():
            raise ValueError(f"Training sample {item} has no usable '{target_key}' target")
        row["hpt_target"] = target_text
        return row
