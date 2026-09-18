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
# export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1


export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export HCCL_CONNECT_TIMEOUT=1500

unset CUDA_VISIBLE_DEVICES || true

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline # add 
export VLLM_USE_V1=1
export DEVICE_TYPE=npu
export VLLM_ASCEND_ENABLE_NZ=0
export TORCHINDUCTOR_DYNAMIC_SHAPES=0

export OUTPUT_PATH # ADD "/home/ma-user/modelarts/outputs/output_path_0/"

export WANDB_PROJECT="luffy-srft-dft"
export PYTHONPATH=$WORK_DIR/verl/hpt:$WORK_DIR/verl/hpt/verl:${PYTHONPATH:-}

UNIFY_STRATEGY="switch"
SWITCH_GATE=0
SWITCH_GATE_OFF=0
OFFLINE_LOSS_TYPE="sft"
SFT_LOSS_COEF=1.0 # 0.3 for Qwen2.5-Math-1.5B
REMOVE_SFTED_DATA=False
MAX_GRAD_NORM=80.0

LR=5e-6
MODEL=Qwen2.5-Math-7B-16k
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/Scaf-GRPO/${MODEL}}"

EXP_BASENAME="VERL070_NPU_${DATE}_${UNIFY_STRATEGY}-${OFFLINE_LOSS_TYPE}-${SFT_LOSS_COEF}_${MODEL}_gate@${SWITCH_GATE}_lr@${LR}_${TIME_TAG}"
EXP_NAME="${OUTPUT_PATH}/${EXP_BASENAME}"

export TENSORBOARD_DIR="${EXP_NAME}/tensorboard"

# OUTPUTS_ROOT="$(dirname "${OUTPUT_PATH}")"
# WANDB_OUTPUT_DIR="${EXP_NAME}/wandb"

# CONVERT_SCRIPT="/home/ma-user/work/verl/tools/tensorboard_to_wandb.py"

mkdir -p \
    "${EXP_NAME}" \
    "${TENSORBOARD_DIR}"

exec > >(tee -a "${EXP_NAME}/train.log") 2>&1


printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

reward_tag="math-verify"
prompt_tag="system-p1"

data_dir="${DATA_DIR:-data}"
TRAIN_FILE=${TRAIN_FILE:-"${data_dir}/openr1.1024.4096.sample.parquet"} # openr1.parquet
TEST_FILE=${TEST_FILE:-["${data_dir}/AIME24/test.parquet","${data_dir}/AMC23/test.parquet","${data_dir}/MATH-500/test.parquet"]}

cd $WORK_DIR/verl/hpt/verl/

python3.10 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.seed=47 \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    \
    +actor_rollout_ref.hpt_actor.offline_loss_type="$OFFLINE_LOSS_TYPE" \
    +actor_rollout_ref.hpt_actor.sft_loss_coef=$SFT_LOSS_COEF \
    +actor_rollout_ref.hpt_actor.loss_remove_token_mean=true \
    +actor_rollout_ref.hpt_actor.loss_remove_clip=true \
    +actor_rollout_ref.hpt_actor.max_grad_norm=$MAX_GRAD_NORM \
    +actor_rollout_ref.hpt_actor.use_ppo_kl_loss=false \
    +actor_rollout_ref.hpt_actor.ppo_kl_loss_coef=0.01 \
    +actor_rollout_ref.hpt_actor.use_adaptive_temperature=false \
    +actor_rollout_ref.hpt_actor.use_adaptive_temperature_fixed=false \
    +actor_rollout_ref.hpt_actor.adaptive_temperature_clip=-1 \
    +actor_rollout_ref.hpt_actor.adaptive_temperature_target_entropy=1.0 \
    +actor_rollout_ref.hpt_actor.alpha_lr=1e-2 \
    \
    +trainer.unify_strategy="$UNIFY_STRATEGY" \
    +trainer.switch_gate="$SWITCH_GATE" \
    +trainer.switch_gate_off=$SWITCH_GATE_OFF \
    +trainer.success_value=1.0 \
    +data.target_key=target \
    +data.reward_impl_version=6 \
    \
    reward_model.launch_reward_fn_async=false \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.bypass_mode=false \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_BASENAME" \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$EXP_NAME/checkpoints \
    trainer.use_legacy_worker_impl=enable \
    reward_model.use_reward_loop=false \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    "$@"