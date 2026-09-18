#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HPT_STRATEGY=mix
export OFFLINE_LOSS_TYPE=off_policy
export LR=${LR:-1e-6}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-luffy-qwen}
exec bash "${SCRIPT_DIR}/run_hpt.sh"
