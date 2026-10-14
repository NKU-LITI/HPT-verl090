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
from functools import lru_cache, partial

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.import_utils import load_extern_object
from verl.utils.reward_score import default_compute_score

RESPONSE_LENGTH_KEY = "response_length"


@lru_cache(maxsize=None)
def _load_configured_score(module_path: str, function_name: str):
    """Load the configured scorer inside the spawned reward process."""
    return load_extern_object(module_path=module_path, object_name=function_name)


def _compute_configured_score_in_process(
    module_path: str,
    function_name: str,
    reward_kwargs: dict,
    *,
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    max_response_length: int,
):
    """Invoke the selected custom scorer without pickling the function object.

    File-loaded functions are not reliably pickleable with the ``spawn`` start
    method.  Passing the module specification lets the worker import either the
    origin scorer or the current scorer itself while retaining SIGALRM support.
    """
    compute_score = _load_configured_score(module_path, function_name)
    call_kwargs = {
        "extra_info": extra_info,
        "max_response_length": max_response_length,
    }
    # Match trainer.ppo.reward._call_with_kwargs: configured kwargs win over
    # values supplied by the reward manager.
    call_kwargs.update(reward_kwargs)
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        **call_kwargs,
    )


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
        reward_fn_config = config.reward.get("custom_reward_function") or {}
        self.custom_reward_path = reward_fn_config.get("path")
        self.custom_reward_name = reward_fn_config.get("name")
        self.custom_reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
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
            if self.custom_reward_path and self.custom_reward_name:
                score_call = partial(
                    _compute_configured_score_in_process,
                    self.custom_reward_path,
                    self.custom_reward_name,
                    self.custom_reward_kwargs,
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    max_response_length=response_length,
                )
            else:
                score_call = partial(
                    self.compute_score,
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    max_response_length=response_length,
                    **extra_reward_kwargs,
                )
            result = await self.loop.run_in_executor(self._legacy_score_executor, score_call)

        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info = dict(result)
        else:
            score = result
            reward_extra_info = {"acc": score}
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
