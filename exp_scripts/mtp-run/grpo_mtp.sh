#!/usr/bin/env bash
set -x

# GRPO | Qwen3.5-35B | mm_k12_math | vLLM rollout | FSDP | Ascend 910B multi-node
# 多节点版本：纯 GRPO + 规则奖励（rule-only），无 GenRM
#
# 与 run_qwen3_5_grpo_910b_fsdp_reward_35b_image_test_zsg.sh 的差异：
#   - 删除所有 GRM_* 配置和 env 透传
#   - 使用 HPT legacy entropy_math reward；同步 scorer 必须通过
#     LegacyHPTRewardManager 在独立进程主线程执行，不能回退到 naive 线程池。
#   - trainer 占用全部 NNODES（不再为 GenRM 预留 GRM_NNODES）
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

echo "******************* copy file *************************"
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

python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

# MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-35B-A3B/Qwen/Qwen3___5-35B-A3B} 

NNODES=${NNODES:-$MA_NUM_HOSTS}
EXPERIMENT_NAME=${EXPERIMENT_NAME}

NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

# ================= 规则奖励（rule-only）配置 =================
# 不启用 GenRM，也不启动 reward_model router
# reward 走 verl/utils/reward_score/__init__.py:default_compute_score
# 按 data_source=="mm_k12_math" 分发到 mm_k12_math_penalty_07311944.compute_score
# 若需要切换规则函数（如换用 mm_k12_math_penalty_new），修改 __init__.py 里 mm_k12_math 分支即可
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
# ========================================================================


# GRPO 不需要区分 Student/Teacher，整节点资源全给 actor/rollout/ref
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}
# rule-only：trainer 占用全部 NNODES（不再为 GenRM 预留节点）
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}

# TRAIN_FILE=${TRAIN_FILE:-${cq_qwen_datasets}/rl_datasets/exp_rl_0910_397b/merged_all_rl_data_15k_0912.parquet}
# TEST_FILE=${TEST_FILE:-${cq_qwen_datasets}/exp_train_json/test_rl_datasets_hard_8437_middle_1563_math_no_prompt2_0806.parquet}


################## 下面是我的脚本 ##################
export ACL_OP_EXECUTE_TIMEOUT=3600 # [ADD]

# ROOT=. # $WORK_DIR/verl
export WANDB_PROJECT=${WANDB_PROJECT:-"luffy-srft-dft"}
export PYTHONPATH=hpt:hpt/verl:${PYTHONPATH:-}

UNIFY_STRATEGY="no"
OFFLINE_LOSS_TYPE="grpo"


LR=${LR:-5e-6}
MODEL=${MODEL:-Qwen2.5-Math-7B} # Qwen2.5-Math-7B-16k
# MODEL_PATH=${MODEL_PATH:-"/home/ma-user/work/Scaf-GRPO/${MODEL}"} 
MODEL_PATH=${MODEL_PATH:-"${upload_zyz}/${MODEL}"} # TODO
EXP_BASENAME=${EXP_BASENAME:-"VERL090_MTP_${DATE}_GRPO_${MODEL}_lr@${LR}_${TIME_TAG}"}

OUTPUT_PATH=${OUTPUT_PATH:-${zlt_data}/qwen25_math_7b/GRPO}
EXP_NAME="${OUTPUT_PATH}/${EXP_BASENAME}" # OUTPUT_PATH


LOG_PATH=${LOG_PATH:-${EXP_NAME}/}


export TENSORBOARD_DIR="${EXP_NAME}/tensorboard"

mkdir -p \
    "${EXP_NAME}" \
    "${TENSORBOARD_DIR}"

exec > >(tee -a "${EXP_NAME}/train.log") 2>&1

printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

DATA_DIR=${DATA_DIR:-"${zlt_data}/HPT_data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.1024.4096.sample.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_DIR}/AIME24/test.parquet","${DATA_DIR}/AMC23/test.parquet","${DATA_DIR}/MATH-500/test.parquet"]}



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

export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000
echo $RAY_gcs_rpc_server_reconnect_timeout_s



############## END #####################

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

            cd hpt/verl/
            
            python3 -m verl.trainer.main_ppo \
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
                actor_rollout_ref.model.path=$MODEL_PATH \
                actor_rollout_ref.model.use_remove_padding=False \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                +actor_rollout_ref.model.override_config.attn_implementation=eager \
                actor_rollout_ref.actor.optim.lr=$LR \
                actor_rollout_ref.actor.ppo_mini_batch_size=32 \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_dynamic_bsz=False \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
                actor_rollout_ref.actor.policy_loss.loss_mode=grpo_compat \
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
                reward.custom_reward_function.path="hpt/verl/verl/experimental/hpt/reward.py" \
                reward.custom_reward_function.name=compute_score \
                reward.reward_manager.source=importlib \
                reward.reward_manager.name=LegacyHPTRewardManager \
                reward.reward_manager.module.path="hpt/verl/verl/experimental/hpt/reward_manager.py" \
                trainer.logger=['console','tensorboard'] \
                trainer.project_name="$WANDB_PROJECT" \
                trainer.experiment_name="$EXP_BASENAME" \
                trainer.use_v1=False \
                trainer.val_before_train=False \
                trainer.nnodes=1 \
                trainer.save_freq=50 \
                trainer.test_freq=50 \
                trainer.max_actor_ckpt_to_keep=2 \
                data.shuffle=True \
                trainer.total_epochs=1 \
                trainer.default_local_dir=$EXP_NAME/checkpoints \
                trainer.resume_mode=auto 2>&1 | tee -a ${LOG_PATH}/${SAVELOG_FILE_NAME}
            python_exit_code=${PIPESTATUS[0]}
            if [ "${python_exit_code}" -ne 0 ]; then
                echo "Training failed with exit code ${python_exit_code}...."
                sleep 3h
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
