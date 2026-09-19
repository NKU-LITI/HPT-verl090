#!/usr/bin/env bash

set -x
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)

# =========================
# Ascend environment
# =========================

source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi

# 使用全部 8 张 910B3
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}

unset CUDA_VISIBLE_DEVICES || true

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_USE_V1=1
export DEVICE_TYPE=npu
export VLLM_ASCEND_ENABLE_NZ=0
export TORCHINDUCTOR_DYNAMIC_SHAPES=0

ROOT=$WORK_DIR/verl
OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/outputs}
export WANDB_PROJECT=${WANDB_PROJECT:-"luffy-srft-dft"}
export PYTHONPATH=$ROOT/hpt:$ROOT/hpt/verl:${PYTHONPATH:-}

UNIFY_STRATEGY="mix"
SWITCH_GATE=${SWITCH_GATE:-0}
OFFLINE_LOSS_TYPE=${OFFLINE_LOSS_TYPE:-off_policy}
SFT_LOSS_COEF=${SFT_LOSS_COEF:-1.0}
OFF_POLICY_BETA=${OFF_POLICY_BETA:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-80.0}

LR=${LR:-1e-6}
MODEL=${MODEL:-Qwen2.5-Math-7B-16k}
MODEL_PATH=${MODEL_PATH:-"/home/ma-user/work/Scaf-GRPO/${MODEL}"}
EXP_BASENAME=${EXP_BASENAME:-"VERL090_NPU_${DATE}_LUFFY_${MODEL}_lr@${LR}_${TIME_TAG}"}
EXP_NAME="${OUTPUT_PATH}/${EXP_BASENAME}"

export TENSORBOARD_DIR="${EXP_NAME}/tensorboard"

mkdir -p \
    "${EXP_NAME}" \
    "${TENSORBOARD_DIR}"

exec > >(tee -a "${EXP_NAME}/train.log") 2>&1

printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

DATA_DIR=${DATA_DIR:-"$ROOT/data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.1024.4096.sample.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_DIR}/AIME24/test.parquet","${DATA_DIR}/AMC23/test.parquet","${DATA_DIR}/MATH-500/test.parquet"]}

cd $ROOT/hpt/verl/

python3.10 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    trainer.n_gpus_per_node=8 \
    trainer.device=npu \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.seed=47 \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.custom_cls.path="$ROOT/hpt/verl/verl/experimental/hpt/dataset.py" \
    data.custom_cls.name=HPTDataset \
    +data.target_key=target \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.policy_loss.loss_mode=hpt \
    actor_rollout_ref.actor.policy_loss.loss_remove_token_mean=true \
    actor_rollout_ref.actor.policy_loss.loss_remove_clip=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="$ROOT/hpt/verl/verl/experimental/hpt/reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=LegacyHPTRewardManager \
    reward.reward_manager.module.path="$ROOT/hpt/verl/verl/experimental/hpt/reward_manager.py" \
    +hpt.enabled=True \
    +hpt.strategy="$UNIFY_STRATEGY" \
    +hpt.switch_gate=$SWITCH_GATE \
    +hpt.success_reward=1.0 \
    +hpt.offline_loss_type="$OFFLINE_LOSS_TYPE" \
    +hpt.sft_loss_coef=$SFT_LOSS_COEF \
    +hpt.off_policy_beta=$OFF_POLICY_BETA \
    +hpt.max_grad_norm=$MAX_GRAD_NORM \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_BASENAME" \
    trainer.use_v1=False \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    data.shuffle=True \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$EXP_NAME/checkpoints \
    trainer.resume_mode=auto \
    "$@"
