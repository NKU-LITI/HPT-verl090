#!/bin/bash
set -ex

PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTPS_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"

echo "************************************** copy algorithm **************************************"

ROOT_PATH="/opt/huawei"
PROJECT_ROOT=/cache/algorithm
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}

ls ${PROJECT_ROOT}

DATA_BASEPATH=${zlt_data}

echo "***************************************** set huawei env *****************************************"

cd ${PROJECT_ROOT}
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
source /home/ma-user/Ascend/nnal/atb/set_env.sh

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=
unset PYTORCH_NPU_ALLOC_CONF

export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

export MODEL_TYPE="qwen3_5_moe"

NNODES=${NNODES:-$MA_NUM_HOSTS}
NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

### 分布式网络配置（多节点）
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=${MASTER_PORT:-6167}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=/cache/ray_tmp
mkdir -p /cache/ray_tmp

export NPUS_PER_NODE=8
export NPROC_PER_NODE=${NPUS_PER_NODE}
export WORLD_SIZE=$(( NNODES * NPUS_PER_NODE ))

readonly SOCKET_IFNAME="eth0"

export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME

export VLLM_RPC_TIMEOUT=600000

echo "***************************************** set custom env *****************************************"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

MODEL='Qwen3.5-35B-A3B'

LOSS_TYPE=${LOSS_TYPE:-sft}
LOSS_TYPE=${LOSS_TYPE,,}
if [[ "${LOSS_TYPE}" != "sft" && "${LOSS_TYPE}" != "dft" ]]; then
    echo "LOSS_TYPE must be either 'sft' or 'dft', got '${LOSS_TYPE}'." >&2
    exit 2
fi

LR=${LR:-5e-6}

ENV_EXP_NAME=${ENV_EXP_NAME:-1}
EXP_NAME=${EXP_NAME:-"VERL090_${LOSS_TYPE^^}_${MODEL}_lr@${LR}_${ENV_EXP_NAME}"}

MODEL_PATH=${edu_gy_zsg}qwen35_rl/openllm_model/Qwen/Qwen3___5-35B-A3B

DATA_DIR=${DATA_DIR:-"${DATA_BASEPATH}/HPT_data"}

# TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.1024.4096.sample.parquet"}
# TEST_FILE=${TEST_FILE:-'${DATA_DIR}/AIME24/test.parquet','${DATA_DIR}/AMC23/test.parquet','${DATA_DIR}/MATH-500/test.parquet'}
TEST_FREQ=${TEST_FREQ:--1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
let MAX_LENGTH=MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH

PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
PPO_TOKEN_LEN_PER_GPU=${PPO_TOKEN_LEN_PER_GPU:-8192}

MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
TOKEN_LEN_PER_GPU=${TOKEN_LEN_PER_GPU:-8192}

MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}

ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

SAVE_DIR=${DATA_BASEPATH}/checkpoints/${EXP_NAME}
LOG_DIR=${DATA_BASEPATH}/logs/${EXP_NAME}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${DATA_BASEPATH}/rollout/${EXP_NAME}}
mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${ROLLOUT_DATA_DIR}"

echo "LOG_PATH resolved to: ${LOG_DIR}"
echo "ROLLOUT_DATA_DIR resolved to: ${ROLLOUT_DATA_DIR}"

TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_FREQ=${VAL_FREQ:-10}

# MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-3200}

# ---

SP_SIZE=1
DP_SIZE=$((NPROC_PER_NODE / SP_SIZE))

if (( SP_SIZE < 1 || NPROC_PER_NODE % SP_SIZE != 0 )); then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by SP_SIZE (${SP_SIZE})." >&2
    exit 2
fi
if (( TRAIN_BATCH_SIZE % DP_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be divisible by data-parallel size (${DP_SIZE})." >&2
    exit 2
fi

python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

echo "***************************************** install verl packages *****************************************"

cd ${PROJECT_ROOT}/hpt/verl
pip install -e . --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ${PROJECT_ROOT}

pip install transformers==5.10.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install aiohttp aiofiles math-verify -i https://pypi.tuna.tsinghua.edu.cn/simple

which python
python -V
pip list

cd ${PROJECT_ROOT}
echo "***************************************** fresh-clean *****************************************"
echo "Waiting 20s for NPU to release resources..."
sleep 20

echo "***************************************** start training with torchrun *****************************************"
cd ${PROJECT_ROOT}/hpt/verl/

python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.sft_trainer \
    data.train_files="['${DATA_DIR}/math.parquet']" \
    data.val_files=null \
    data.messages_key=null \
    data.prompt_key=prompt \
    data.target_key=target \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.max_length=$MAX_LENGTH \
    data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.ignore_input_ids_mismatch=False \
    model.path=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=True \
    engine.strategy=fsdp2 \
    engine.param_offload=True \
    engine.optimizer_offload=True \
    engine.ulysses_sequence_parallel_size=$SP_SIZE \
    optim=fsdp \
    optim.lr=$LR \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    trainer.logger=['console','tensorboard'] \
    trainer.loss_type=$LOSS_TYPE \
    trainer.experiment_name="$EXP_NAME" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=$TEST_FREQ \
    trainer.max_ckpt_to_keep=2 \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.resume_mode=auto \
    "$@" 2>&1 | tee -a ${LOG_DIR}/${LOSS_TYPE}_${NODE_RANK}.log

python_exit_code=${PIPESTATUS[0]}
if [ "${python_exit_code}" -ne 0 ]; then
    echo "Training failed with exit code ${python_exit_code}...."
    sleep 3h
    exit ${python_exit_code}
fi
