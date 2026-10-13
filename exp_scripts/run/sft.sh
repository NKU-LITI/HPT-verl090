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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-online}

source /home/liting/miniconda3/etc/profile.d/conda.sh
conda activate verl090
# ------------------------------------------------------------------------------------------------

export WANDB_PROJECT=${WANDB_PROJECT:-"luffy-srft-dft"}

LOSS_TYPE=${LOSS_TYPE:-sft}
LR=${LR:-5e-6}
MODEL=${MODEL:-Qwen2.5-Math-7B} # rope_theta=40000 and max_position_embeddings=16384
EXP_NAME=${EXP_NAME:-"VERL090_${DATE}_${LOSS_TYPE^^}_${MODEL}_lr@${LR}_${TIME_TAG}"}
MODEL_PATH=${MODEL_PATH:-/workplace/nankai/liting_space/LLM/$MODEL}
DATA_DIR=${DATA_DIR:-$ROOT/data}

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
SP_SIZE=${SP_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-6144}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

cd $ROOT/hpt/verl/
mkdir -p $ROOT/checkpoints/$EXP_NAME

TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.1024.4096.sample.parquet"}
# The benchmark test files have no assistant target; pass a supervised SFT file to enable validation.
TEST_FILE=${TEST_FILE:-null}
TEST_FREQ=${TEST_FREQ:--1}

if (( SP_SIZE < 1 || NPROC_PER_NODE % SP_SIZE != 0 )); then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by SP_SIZE (${SP_SIZE})." >&2
    exit 2
fi
DP_SIZE=$((NPROC_PER_NODE / SP_SIZE))
if (( TRAIN_BATCH_SIZE % DP_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be divisible by data-parallel size (${DP_SIZE})." >&2
    exit 2
fi

python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.messages_key=messages \
    data.prompt_key=prompt \
    data.target_key=target \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.max_length=$MAX_LENGTH \
    data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.ignore_input_ids_mismatch=False \
    model.path=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=True \
    engine=fsdp \
    engine.strategy=fsdp \
    engine.ulysses_sequence_parallel_size=$SP_SIZE \
    optim=fsdp \
    optim.lr=$LR \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    trainer.logger=['console','wandb'] \
    trainer.loss_type=$LOSS_TYPE \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=$TEST_FREQ \
    trainer.max_ckpt_to_keep=2 \
    trainer.default_local_dir=$ROOT/checkpoints/$EXP_NAME \
    trainer.resume_mode=auto \
    "$@"
