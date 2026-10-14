#!/bin/bash
set -x

# GRPO | Qwen3.5-4B | Geo3K | vLLM rollout | FSDP | Ascend 910B multi-node
# 多节点版本：纯 GRPO，无 Teacher/Distillation
#
# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)
echo "datetime: $(date +'%Y-%m-%d %H:%M:%S')"
echo $MA_CURRENT_HOST_IP
echo $MA_CURRENT_IP

################## 上海三方电信 代理配置 ##################
PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTPS_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"

echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
PROJECT_ROOT=/cache/algorithm

ROOT_PATH="/opt/huawei"
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
ls ${PROJECT_ROOT}

# cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-toolkit_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
# cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-910b-ops_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
# cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-nnal_8.5.2_linux-aarch64.run ${PROJECT_ROOT}

# ls ${PROJECT_ROOT}
echo "***************************************** update cann *****************************************"
ls /home/ma-user/
ls /home/ma-user/Ascend/

# cd ${PROJECT_ROOT}
# chmod +777 ./Ascend-cann-*
# ./Ascend-cann-toolkit_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
# ./Ascend-cann-910b-ops_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
# ./Ascend-cann-nnal_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/nnal/atb/set_env.sh

echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF

export ACL_OP_COMPILER_CACHE_DIR=$ACL_OP_COMPILER_CACHE_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

# echo "***************************************** Installing Pytorch *****************************************"
# python3 -m pip install --upgrade pip
# pip uninstall jsonnet -y || true
# pip install --upgrade pip

# cd ${PROJECT_ROOT}
# pip install TransferQueue
# pip install transferqueue-0.1.9.dev0-py3-none-any.whl
# pip install torch==2.9.0
# pip install torch_npu==2.9.0
# pip uninstall torch_c_dlpack_ext -y || true
# pip install triton_ascend==3.2.0
# pip install torch==2.9.0
# pip install torch_npu==2.9.0
python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"


# ================= 数据集 & 奖励函数路径（xyi） =================

BASE_PATH=${BASE_PATH:-${zlt_data}}
echo "BASE_PATH resolved to: ${BASE_PATH}"

DATA_DIR=${DATA_DIR:-"${BASE_PATH}/HPT_data"}
# TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.1024.4096.sample.parquet"}
# TRAIN_FILE=${TRAIN_FILE:-"${BASE_PATH}/xyi_data/math_problem_no_length_overflow.parquet"} # xyi.parquet
TRAIN_FILE=${TRAIN_FILE:-"${BASE_PATH}/xyi_data/xyi_2_16_1013.parquet"} # xyi.parquet # math_problem_no_length_overflow

TEST_FILE=${TEST_FILE:-["${DATA_DIR}/AIME24/test.parquet","${DATA_DIR}/AMC23/test.parquet"]}

pip install "accelerate>=1.6.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import accelerate; print(accelerate.__version__)"

pip install latex2sympy2_extended -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import latex2sympy2_extended; print('ok')"

pip install math_verify -i https://pypi.tuna.tsinghua.edu.cn/simple
# ===============================================================

MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-35B-A3B/Qwen/Qwen3___5-35B-A3B} 

NNODES=${NNODES:-$MA_NUM_HOSTS}
EXPERIMENT_NAME=${EXPERIMENT_NAME}


# ================= TensorBoard 输出目录（带时间戳） =================
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
PROJECT_NAME=${PROJECT_NAME:-VERL090_MTP_${DATE}_HPT_ORIGIN_REWARD_Qwen35-35B-A3B_lr@${LR}_${TIME_TAG}_${TIMESTAMP}}


TENSORBOARD_DIR="${BASE_PATH}/tensorboard/${PROJECT_NAME}/"
mkdir -p "${TENSORBOARD_DIR}"
echo "TENSORBOARD_DIR = ${TENSORBOARD_DIR}"
# ====================================================================

NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

# ================= 规则奖励（rule-only）配置 =================
# 不启用 GenRM；由 LegacyHPTRewardManager 在独立进程中调用 origin reward。
# ========================================================================


# GRPO 不需要区分 Student/Teacher，整节点资源全给 actor/rollout/ref
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}
# rule-only：trainer 占用全部 NNODES（不再为 GenRM 预留节点）
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}



# TRAIN_BATCH_SIZE 和 PPO_MINI_BATCH_SIZE设置成卡数
# train_batch_size * 卡数 得是 mini_batch_size的整数倍
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-192}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192} 
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12888}

ACTOR_LR=${ACTOR_LR:-5e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}

ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${BASE_PATH}/ckpt/${PROJECT_NAME}/}

LOG_PATH=${LOG_PATH:-${BASE_PATH}/log/${PROJECT_NAME}/}
echo "LOG_PATH resolved to: ${LOG_PATH}"

ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${BASE_PATH}/rollout/${PROJECT_NAME}/}
echo "ROLLOUT_DATA_DIR resolved to: ${ROLLOUT_DATA_DIR}"


TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
VAL_FREQ=${VAL_FREQ:-10}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-12888}


# [HPT]
UNIFY_STRATEGY="switch"
SWITCH_GATE=${SWITCH_GATE:-0}
OFFLINE_LOSS_TYPE=${OFFLINE_LOSS_TYPE:-sft}
SFT_LOSS_COEF=${SFT_LOSS_COEF:-1.0}
OFF_POLICY_BETA=${OFF_POLICY_BETA:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-80.0}



### 启动分布式网络配置（多节点）
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=/cache/ray_tmp
mkdir -p /cache/ray_tmp

export NPUS_PER_NODE=8
export WORLD_SIZE=$(( NNODES * NPUS_PER_NODE ))

readonly SOCKET_IFNAME="eth0"

export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export ASCEND_GLOBAL_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export VLLM_RPC_TIMEOUT=600000

pip list

export TENSORBOARD_DIR="${TENSORBOARD_DIR}"

export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000

echo $RAY_gcs_rpc_server_reconnect_timeout_s

npu-smi info

# ================= 多节点：启动 Ray 集群 =================
if [ "${NODE_RANK}" = "0" ]; then
    # ---- 头节点：启动 Ray Head ----
    echo "Starting Ray HEAD node on ${MA_CURRENT_IP}..."
    ray start --head \
        --node-ip-address=${MA_CURRENT_IP} \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --resources='{"NPU": '${NPUS_PER_NODE}'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / NPUS_PER_NODE))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster ready: ${device_count} nodes"
            ray status

            mkdir -p ${LOG_PATH}

            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                algorithm.norm_adv_by_std_in_grpo=False \
                algorithm.use_kl_in_reward=False \
                critic.enable=False \
                data.train_files=$TRAIN_FILE \
                data.val_files=$TEST_FILE \
                data.image_key=images \
                data.prompt_key=prompt \
                data.reward_fn_key=data_source \
                data.return_raw_chat=True \
                data.custom_cls.path="pkg://verl.experimental.hpt.dataset" \
                data.custom_cls.name=HPTDataset \
                +data.target_key=target \
                data.train_batch_size=${TRAIN_BATCH_SIZE} \
                data.max_prompt_length=${MAX_PROMPT_LENGTH} \
                data.max_response_length=${MAX_RESPONSE_LENGTH} \
                data.filter_overlong_prompts=True \
                data.truncation='error' \
                data.shuffle=True \
                data.seed=47 \
                actor_rollout_ref.model.path="${MODEL_PATH}" \
                actor_rollout_ref.model.use_remove_padding=False \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.strategy=fsdp2 \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
                actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
                actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
                actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
                actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_dynamic_bsz=False \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.01 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
                actor_rollout_ref.actor.calculate_entropy=False \
                actor_rollout_ref.actor.policy_loss.loss_mode=hpt \
                actor_rollout_ref.actor.policy_loss.loss_remove_token_mean=True \
                actor_rollout_ref.actor.policy_loss.loss_remove_clip=True \
                actor_rollout_ref.actor.use_kl_loss=False \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']" \
                actor_rollout_ref.ref.strategy=fsdp2 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
                actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
                actor_rollout_ref.rollout.n=${ROLLOUT_N} \
                actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
                actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
                actor_rollout_ref.rollout.val_kwargs.n=1 \
                actor_rollout_ref.rollout.val_kwargs.do_sample=True \
                actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
                actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
                actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
                actor_rollout_ref.rollout.enforce_eager=True \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.rollout.enable_chunked_prefill=False \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
                reward.custom_reward_function.path="pkg://verl.experimental.hpt.reward_origin" \
                reward.custom_reward_function.name=compute_score \
                reward.reward_manager.source=importlib \
                reward.reward_manager.name=LegacyHPTRewardManager \
                reward.reward_manager.module.path="pkg://verl.experimental.hpt.reward_manager_origin" \
                +hpt.enabled=True \
                +hpt.strategy="$UNIFY_STRATEGY" \
                +hpt.switch_gate=$SWITCH_GATE \
                +hpt.success_reward=1.0 \
                +hpt.offline_loss_type="$OFFLINE_LOSS_TYPE" \
                +hpt.sft_loss_coef=$SFT_LOSS_COEF \
                +hpt.off_policy_beta=$OFF_POLICY_BETA \
                +hpt.max_grad_norm=$MAX_GRAD_NORM \
                trainer.balance_batch=True \
                trainer.logger='["console","tensorboard"]' \
                trainer.project_name=${PROJECT_NAME} \
                trainer.experiment_name=${EXPERIMENT_NAME} \
                trainer.n_gpus_per_node=${NTRAINER_GPUS_PER_NODE} \
                trainer.nnodes=${TRAINER_NNODES} \
                trainer.device=npu \
                trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
                trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
                trainer.val_before_train=False \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=${VAL_FREQ} \
                trainer.total_epochs=${TOTAL_EPOCHS} 2>&1 | tee -a ${LOG_PATH}/${SAVELOG_FILE_NAME}
            python_exit_code=${PIPESTATUS[0]}
            if [ "${python_exit_code}" -ne 0 ]; then
                echo "Training failed with exit code ${python_exit_code}...."
                sleep 30h
                exit ${python_exit_code}
            fi
            break
        fi

        echo "Waiting for Ray nodes. Current=${device_count}, expected=${NNODES}"
        sleep 10
    done
else
    echo "Ray Head is up! Starting worker..."

    ray start \
    --node-ip-address=${MA_CURRENT_IP} \
    --address=${MASTER_ADDR}:${MASTER_PORT} \
    --resources='{"NPU": '${NPUS_PER_NODE}'}'

    sleep infinity
fi
