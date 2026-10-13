# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reward-manager compatibility for canonical verl 0.1 HPT."""

import inspect
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from verl import DataProto
from verl.experimental.hpt.reward import RESPONSE_LENGTH_KEY, compute_score_in_process
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


def decode_legacy_response(tokenizer, response_ids) -> str:
    """The 0.1 HPT RewardManager retains special tokens while decoding."""
    return tokenizer.decode(response_ids)


class LegacyHPTRewardManager(RewardManagerBase):
    """Naive reward manager with the exact 0.1 response decode contract."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        # verl 0.1 scored serially in the process main thread. verl 0.9's
        # reward loop sends synchronous functions to a ThreadPoolExecutor,
        # where entropy_math's SIGALRM guards either fail closed or must be
        # disabled (which can hang indefinitely in SymPy). A persistent spawned
        # worker restores both the legacy score and its hard timeout semantics.
        self._legacy_score_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))

    async def run_single(self, data: DataProto) -> dict:
        data_item = data[-1:][0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None, lambda: decode_legacy_response(self.tokenizer, valid_response_ids)
        )
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        # The same dict can be shared by all rollouts of one prompt.
        extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info[RESPONSE_LENGTH_KEY] = valid_response_length

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                max_response_length=response_length,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                self._legacy_score_executor,
                partial(
                    compute_score_in_process,
                    data_source,
                    response_str,
                    ground_truth,
                    extra_info,
                    max_response_length=response_length,
                ),
            )

        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info = dict(result)
        else:
            score = result
            reward_extra_info = {"acc": score}
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
