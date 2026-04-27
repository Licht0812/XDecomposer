#!/bin/bash
#SBATCH -J transformer_sep
#SBATCH -p project1
#SBATCH -A project1
#SBATCH -o logs/transformer_%j.out
#SBATCH -e logs/transformer_%j.err
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    SCRIPT_DIR="$PROJECT_ROOT/baselines/transformer_family"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

source "$PROJECT_ROOT/configs/paths.sh"
source "$SCRIPT_DIR/env_utils.sh"

# Launcher defaults. Training hyperparameters live in train_config.yaml.
BASELINE="transformer"
GPUS="${CUDA_VISIBLE_DEVICES:-}"
PORT=$((29500 + RANDOM % 100))
RUN_NAME="separation"
CONFIG_PATH="${PATH_FILE_BASELINE_TRAIN_CONFIG:-$SCRIPT_DIR/train_config.yaml}"
RESUME_PATH=""
EXTRA_ARGS=()

require_option_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == -* ]]; then
        echo "Missing value for $option" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline|--model) require_option_value "$1" "${2:-}"; BASELINE="$2"; shift 2 ;;
        --gpus) require_option_value "$1" "${2:-}"; GPUS="$2"; shift 2 ;;
        --port) require_option_value "$1" "${2:-}"; PORT="$2"; shift 2 ;;
        --name|--run-name) require_option_value "$1" "${2:-}"; RUN_NAME="$2"; shift 2 ;;
        --config) require_option_value "$1" "${2:-}"; CONFIG_PATH="$2"; shift 2 ;;
        --resume) require_option_value "$1" "${2:-}"; RESUME_PATH="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            if [[ "$1" == -* ]]; then
                EXTRA_ARGS+=("$1")
                shift
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    EXTRA_ARGS+=("$1")
                    shift
                done
                continue
            elif [ -z "$RESUME_PATH" ]; then
                RESUME_PATH="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

activate_spectra_env
cd "$PROJECT_ROOT"
mkdir -p logs

configure_gpu_layout
RESUME_ARG=()
if [ -n "$RESUME_PATH" ]; then
    CLEAN_PATH="$RESUME_PATH"
    if [[ "$RESUME_PATH" == *".ptbash" ]]; then
        CLEAN_PATH="${RESUME_PATH%bash}"
    fi
    RESUME_ARG=(--resume "$CLEAN_PATH")
fi

export PYTHONPATH=.
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-XRD-Transformer-Baselines}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
export http_proxy="${http_proxy:-http://10.36.204.1:3128}"
export https_proxy="${https_proxy:-http://10.36.204.1:3128}"
export ftp_proxy="${ftp_proxy:-http://10.36.204.1:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export FTP_PROXY="${FTP_PROXY:-$ftp_proxy}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ibp20s0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

configure_distributed_env
TRAIN_ARGS=(
    baselines/transformer_family/train.py
    --config "$CONFIG_PATH"
    --baseline_name "$BASELINE"
    --run_name "$RUN_NAME"
    "${RESUME_ARG[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "============================================"
echo "Baseline: $BASELINE"
echo "Run name: $RUN_NAME"
echo "Nodes: $NNODES"
if [ -z "${SLURM_JOB_NUM_NODES:-}" ]; then
    echo "GPUs: $GPUS (Count: $NUM_GPUS)"
else
    echo "GPUs per node: $NUM_GPUS"
fi
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Config: $CONFIG_PATH"
if [ -n "$RESUME_PATH" ]; then
    echo "Resume: $RESUME_PATH"
fi
echo "SwanLab project: $SWANLAB_PROJECT"
echo "============================================"

if declare -F run_cuda_preflight >/dev/null 2>&1; then
    run_cuda_preflight
else
    echo "CUDA preflight skipped: run_cuda_preflight is not defined. Sync env_utils.sh to enable it." >&2
fi
launch_torchrun "${TRAIN_ARGS[@]}"
