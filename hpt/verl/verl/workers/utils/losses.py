# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def _supervised_loss(model_output, data: TensorDict, loss_type: str):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
        token_log_prob = log_prob_flatten
        token_mask = loss_mask_flatten
    else:
        token_log_prob = log_prob
        token_mask = data["response_mask"].to(bool)

    if loss_type == "sft":
        token_loss = -token_log_prob
    elif loss_type == "dft":
        # DFT reweights each target token with the current model probability.
        # The weight must not receive gradients; otherwise this is an entropy
        # objective instead of the DFT probability-weighted NLL.
        token_loss = -torch.exp(token_log_prob).detach() * token_log_prob
    else:
        raise ValueError(f"Unsupported supervised loss type {loss_type!r}; expected 'sft' or 'dft'")

    # NOTE: loss is averaged over all tokens in the batch across all data parallel groups.
    # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
    # the loss should be scaled by `num_microbatches` for pp schedule.
    loss = masked_sum(token_loss, token_mask) / batch_num_tokens * dp_size

    return loss, {}


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute the standard supervised negative log-likelihood objective."""
    return _supervised_loss(model_output=model_output, data=data, loss_type="sft")


def dft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute DFT's stop-gradient probability-weighted NLL objective."""
    return _supervised_loss(model_output=model_output, data=data, loss_type="dft")


def get_supervised_loss_fn(loss_type: str):
    """Return the configured pure supervised-training objective."""
    loss_type = str(loss_type).lower()
    loss_fns = {"sft": sft_loss, "dft": dft_loss}
    if loss_type not in loss_fns:
        raise ValueError(f"Unsupported trainer.loss_type {loss_type!r}; expected one of {sorted(loss_fns)}")
    return loss_fns[loss_type]


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    hpt_gradient_accumulation = tu.get_non_tensor_data(data=data, key="hpt_legacy_gradient_accumulation", default=None)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # The legacy HPT actor logged the mean of the independently normalized
    # micro-batch objectives.  SUM is correct for stock verl losses that are
    # normalized using global_batch_info, but HPT deliberately ignores that
    # normalization to reproduce the 0.1 objective.  Using SUM here makes the
    # displayed loss depend on the dynamic micro-batch packing count even
    # though the backward objective is unchanged.
    policy_metric_aggregation = AggregationType.MEAN if loss_mode == "hpt" else metric_aggregation

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    for hpt_key in (
        "hpt_sft_mask",
        "hpt_valid_mask",
        "hpt_sft_weight",
        "hpt_offline_mode",
        "hpt_off_policy_beta",
        "hpt_correct_mask",
    ):
        if hpt_key in data:
            fields.append(hpt_key)
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    if "hpt_valid_mask" in data:
        response_mask = response_mask & data["hpt_valid_mask"].bool().unsqueeze(-1)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    hpt_kwargs = {}
    if loss_mode == "hpt":
        hpt_kwargs = {
            "hpt_sft_mask": data.get("hpt_sft_mask"),
            "hpt_valid_mask": data.get("hpt_valid_mask"),
            "hpt_sft_weight": data.get("hpt_sft_weight"),
            "hpt_offline_mode": data.get("hpt_offline_mode"),
            "hpt_off_policy_beta": data.get("hpt_off_policy_beta"),
            "hpt_correct_mask": data.get("hpt_correct_mask"),
            "entropy": entropy,
        }
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        **hpt_kwargs,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=policy_metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        if loss_mode == "hpt":
            # The legacy HPT actor normalized entropy inside each current
            # micro-batch, independently of the PPO mini-batch token count.
            from verl.experimental.hpt.loss import compute_hpt_legacy_entropy_loss

            entropy_loss = compute_hpt_legacy_entropy_loss(entropy, response_mask)
        else:
            entropy_loss = agg_loss(
                loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
            )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=policy_metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    if loss_mode == "hpt" and hpt_gradient_accumulation is not None:
        # 0.1 logged ``policy_loss / gradient_accumulation`` as actor/loss.
        # Keep this as a transient metric; RayPPOTrainer promotes it to the
        # standard ``actor/loss`` key after discarding the engine's summed
        # diagnostic loss.  This does not participate in backward.
        metrics["hpt/legacy_actor_loss"] = Metric(
            value=policy_loss / hpt_gradient_accumulation,
            aggregation=AggregationType.MEAN,
        )

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics
