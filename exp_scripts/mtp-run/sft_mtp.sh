#!/usr/bin/env bash

set -euo pipefail
set -x

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# MTP jobs expose the submitted source under this directory. Copy it to the
# local cache when available, then locate the repository in either location.
PROJECT_ROOT=${PROJECT_ROOT:-/cache/algorithm}
SOURCE_ROOT=${SOURCE_ROOT:-/opt/huawei/schedule-train/algorithm}
if [[ "${COPY_PROJECT:-1}" == "1" && -d "${SOURCE_ROOT}" ]]; then
    mkdir -p "${PROJECT_ROOT}"
    cp -r "${SOURCE_ROOT}/." "${PROJECT_ROOT}/"
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${ROOT:-}" ]]; then
    for candidate in \
        "${WORK_DIR:-}/HPT-verl090" \
        "${PROJECT_ROOT}/HPT-verl090" \
        "$(cd -- "${SCRIPT_DIR}/../.." && pwd)"; do
        if [[ -d "${candidate}/hpt/verl" ]]; then
            ROOT=${candidate}
            break
        fi
    done
fi
if [[ -z "${ROOT:-}" || ! -d "${ROOT}/hpt/verl" ]]; then
    echo "Cannot locate HPT-verl090. Set ROOT to the repository directory." >&2
    exit 2
fi

if [[ -f /home/ma-user/Ascend/cann-8.5.2/set_env.sh ]]; then
    set +u
    source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
    [[ -f /home/ma-user/Ascend/nnal/atb/set_env.sh ]] && source /home/ma-user/Ascend/nnal/atb/set_env.sh
    set -u
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi

export PYTHONPATH="${ROOT}/hpt:${ROOT}/hpt/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}
export DEVICE_TYPE=npu
export NON_MEGATRON=true
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-7200}
export ACL_OP_EXECUTE_TIMEOUT=${ACL_OP_EXECUTE_TIMEOUT:-3600}
export TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-3600}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
unset CUDA_VISIBLE_DEVICES || true
unset PYTORCH_NPU_ALLOC_CONF || true

NNODES=${NNODES:-${MA_NUM_HOSTS:-1}}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${MA_NUM_GPUS:-8}}
WORKER_HOSTS=${VC_WORKER_HOSTS:-}
MASTER_ADDR=${MASTER_ADDR:-${WORKER_HOSTS%%,*}}
MASTER_ADDR=${MASTER_ADDR:-${MA_CURRENT_IP:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-6167}
SP_SIZE=${SP_SIZE:-1}

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( SP_SIZE < 1 || WORLD_SIZE % SP_SIZE != 0 )); then
    echo "World size (${WORLD_SIZE}) must be divisible by SP_SIZE (${SP_SIZE})." >&2
    exit 2
fi
DP_SIZE=$((WORLD_SIZE / SP_SIZE))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(( ((64 + DP_SIZE - 1) / DP_SIZE) * DP_SIZE ))}
if (( TRAIN_BATCH_SIZE % DP_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be divisible by data-parallel size (${DP_SIZE})." >&2
    exit 2
fi

MODEL=${MODEL:-Qwen2.5-Math-7B}
MODEL_BASE=${upload_zyz:-${ROOT}/models}
MODEL_PATH=${MODEL_PATH:-${MODEL_BASE}/${MODEL}}
LR=${LR:-5e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-6144}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}

DATA_DIR=${DATA_DIR:-${ROOT}/data}
TRAIN_FILES=${TRAIN_FILES:-${TRAIN_FILE:-${DATA_DIR}/openr1.1024.4096.sample.parquet}}
VAL_FILES=${VAL_FILES:-${VAL_FILE:-null}}

EXP_BASENAME=${EXP_BASENAME:-VERL090_NPU_${DATE}_SFT_${MODEL}_lr@${LR}_${TIME_TAG}}
OUTPUT_BASE=${OUTPUT_PATH:-${zlt_data:-${ROOT}/outputs}/sft}
EXP_NAME=${EXP_NAME:-${OUTPUT_BASE}/${EXP_BASENAME}}
PROJECT_NAME=${PROJECT_NAME:-hpt-sft}
LOGGER=${LOGGER:-"['console','tensorboard']"}
RESUME_MODE=${RESUME_MODE:-auto}
SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}
if [[ "${VAL_FILES}" == "null" || -z "${VAL_FILES}" ]]; then
    TEST_FREQ=${TEST_FREQ:--1}
else
    TEST_FREQ=${TEST_FREQ:-after_each_epoch}
fi

mkdir -p "${EXP_NAME}/checkpoints" "${EXP_NAME}/tensorboard"
export TENSORBOARD_DIR="${EXP_NAME}/tensorboard"
exec > >(tee -a "${EXP_NAME}/train-rank-${NODE_RANK}.log") 2>&1

cd "${ROOT}/hpt/verl"
PYTHON_BIN=${PYTHON_BIN:-python3}
"${PYTHON_BIN}" -c "import torch_npu; assert torch_npu.npu.is_available(); print('NPU available:', True)"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.messages_key=messages \
    data.prompt_key=prompt \
    data.target_key=target \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.ignore_input_ids_mismatch=False \
    model.path="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=True \
    engine=fsdp \
    engine.strategy=fsdp \
    engine.use_torch_compile=False \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    optim=fsdp \
    optim.lr="${LR}" \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    trainer.device=npu \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_BASENAME}" \
    trainer.logger="${LOGGER}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.default_local_dir="${EXP_NAME}/checkpoints" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-2}" \
    "$@"
