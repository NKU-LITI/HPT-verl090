#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${VERL_ROOT}"

# Required inputs. Paths may be local or any URI supported by verl.
: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the HPT parquet containing prompt, target, and reward_model}"
: "${VAL_FILES:?Set VAL_FILES to a parquet path or Hydra list of paths}"

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-8}
LR=${LR:-5e-6}
TOTAL_STEPS=${TOTAL_STEPS:-500}
PROJECT_NAME=${PROJECT_NAME:-hpt-verl090}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-hpt-qwen}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
HPT_STRATEGY=${HPT_STRATEGY:-switch}
SWITCH_GATE=${SWITCH_GATE:-0}
OFFLINE_LOSS_TYPE=${OFFLINE_LOSS_TYPE:-sft}
SFT_LOSS_COEF=${SFT_LOSS_COEF:-1.0}
OFF_POLICY_BETA=${OFF_POLICY_BETA:-0.1}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  critic.enable=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.custom_cls.path="${VERL_ROOT}/verl/experimental/hpt/dataset.py" \
  data.custom_cls.name=HPTDataset \
  +data.target_key=target \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.policy_loss.loss_mode=hpt \
  actor_rollout_ref.actor.policy_loss.loss_remove_clip=true \
  actor_rollout_ref.actor.policy_loss.loss_remove_token_mean=true \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="${VERL_ROOT}/verl/experimental/hpt/reward.py" \
  reward.custom_reward_function.name=compute_score \
  +hpt.enabled=True \
  +hpt.strategy="${HPT_STRATEGY}" \
  +hpt.switch_gate="${SWITCH_GATE}" \
  +hpt.success_reward=1.0 \
  +hpt.offline_loss_type="${OFFLINE_LOSS_TYPE}" \
  +hpt.sft_loss_coef="${SFT_LOSS_COEF}" \
  +hpt.off_policy_beta="${OFF_POLICY_BETA}" \
  +hpt.max_grad_norm=80.0 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.use_v1=False \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.resume_mode=auto
