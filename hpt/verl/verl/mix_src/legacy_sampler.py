# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""verl-0.1 HPT's resumable random sampler, with 0.9 state adapters."""

from typing import Any

import torch


class HPTLegacyRandomSampler(torch.utils.data.Sampler[int]):
    """Preserve the exact seed/permutation/iterator state machine from 0.1.

    ``state_dict`` and ``load_state_dict`` are the only 0.9 additions.  They
    adapt the original ``get_state``/``set_state`` protocol to
    ``StatefulDataLoader`` without changing sample order.
    """

    def __init__(self, data_source, seed: int = 47):
        self.data_source = data_source
        self.seed = int(seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1].item()

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {
            "perm": self.perm.clone(),
            "perm_index": self.perm_index,
            "generator_state": self.generator.get_state(),
            "seed": self.seed,
            "num_samples": self.num_samples,
        }

    def set_state(self, state: dict[str, Any]):
        saved_num_samples = int(state.get("num_samples", len(state["perm"])))
        if saved_num_samples != self.num_samples:
            raise ValueError(
                "Cannot restore HPT sampler after dataset size changed: "
                f"checkpoint has {saved_num_samples} samples, current dataset has {self.num_samples}"
            )
        self.perm = state["perm"].clone()
        self.perm_index = int(state["perm_index"])
        self.generator.set_state(state["generator_state"])

    state_dict = get_state
    load_state_dict = set_state

