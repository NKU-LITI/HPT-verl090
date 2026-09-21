#!/usr/bin/env bash

set -x
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)

# ray stop
# ------------------------------------------------------------------------------------------------
ROOT=/workplace/nankai/liting_space/SFT-RL/HPT-verl090
export PYTHONPATH=$ROOT/hpt:$ROOT/hpt/verl:${PYTHONPATH:-}

# export HF_ENDPOINT=https://hf-mirror.com
# export no_proxy="127.0.0.1,localhost"
# export NO_PROXY="127.0.0.1,localhost"

# vLLM 0.24 no longer reads the old VLLM_ATTENTION_BACKEND variable.  Keep its
# automatic backend selection: forcing FLASHINFER failed the actor-logprob audit.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-online}

source /home/liting/miniconda3/etc/profile.d/conda.sh
conda activate verl090
# ------------------------------------------------------------------------------------------------

export WANDB_PROJECT=${WANDB_PROJECT:-"luffy-srft-dft"}

UNIFY_STRATEGY="mix"
SWITCH_GATE=${SWITCH_GATE:-0}
OFFLINE_LOSS_TYPE=${OFFLINE_LOSS_TYPE:-off_policy}
SFT_LOSS_COEF=${SFT_LOSS_COEF:-1.0}
OFF_POLICY_BETA=${OFF_POLICY_BETA:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-80.0}
HPT_CALCULATE_LOG_PROBS=${HPT_CALCULATE_LOG_PROBS:-false}

LR=${LR:-1e-6}
MODEL=${MODEL:-Qwen2.5-Math-7B} # rope_theta=40000 and max_position_embeddings=16384
EXP_NAME=${EXP_NAME:-"VERL090_${DATE}_LUFFY_${MODEL}_lr@${LR}_${TIME_TAG}"}
MODEL_PATH=${MODEL_PATH:-/workplace/nankai/liting_space/LLM/$MODEL}
DATA_DIR=${DATA_DIR:-$ROOT/data}

cd $ROOT/hpt/verl/
mkdir -p $ROOT/checkpoints/$EXP_NAME

TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.1024.4096.sample.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_DIR}/AIME24/test.parquet","${DATA_DIR}/AMC23/test.parquet","${DATA_DIR}/MATH-500/test.parquet"]}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.policy_loss.loss_mode=hpt \
    actor_rollout_ref.actor.policy_loss.loss_remove_token_mean=true \
    actor_rollout_ref.actor.policy_loss.loss_remove_clip=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    actor_rollout_ref.rollout.seed=0 \
    actor_rollout_ref.rollout.calculate_log_probs=$HPT_CALCULATE_LOG_PROBS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
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
    +hpt.preserve_v01_rollout_scheduling=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.use_v1=False \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    data.shuffle=True \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$ROOT/checkpoints/$EXP_NAME \
    trainer.resume_mode=auto \
    "$@"
