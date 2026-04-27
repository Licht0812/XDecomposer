#!/bin/bash
#SBATCH -J xdecomposer
#SBATCH -o logs/xdecomposer_%j.out
#SBATCH -e logs/xdecomposer_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "$PROJECT_ROOT/configs/paths.sh"
require_conda_env

require_option_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == -* ]]; then
        echo "Missing value for $option" >&2
        exit 1
    fi
}

GPUS="${CUDA_VISIBLE_DEVICES:-}"
PORT=$((29500 + $RANDOM % 100))
EXP_NAME="xdecomposer"
RESUME_PATH=""
NUM_WORKERS="${NUM_WORKERS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            require_option_value "$1" "${2:-}"
            GPUS="$2"
            shift 2
            ;;
        --port)
            require_option_value "$1" "${2:-}"
            PORT="$2"
            shift 2
            ;;
        --name)
            require_option_value "$1" "${2:-}"
            EXP_NAME="$2"
            shift 2
            ;;
        --resume)
            require_option_value "$1" "${2:-}"
            RESUME_PATH="$2"
            shift 2
            ;;
        *)
            if [ -z "$RESUME_PATH" ] && [[ "$1" != -* ]]; then
                RESUME_PATH="$1"
                shift 1
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
    NUM_GPUS="$(detect_num_gpus)"
else
    if [ -n "$GPUS" ]; then
        NUM_GPUS="$(count_csv_items "$GPUS")"
        export CUDA_VISIBLE_DEVICES="$GPUS"
    else
        NUM_GPUS="$(detect_num_gpus)"
    fi
fi

echo "============================================"
echo "Experiment: $EXP_NAME"
if [ -n "$GPUS" ]; then
    echo "GPUs: $GPUS (Count: $NUM_GPUS)"
else
    echo "GPUs: auto-detected (Count: $NUM_GPUS)"
fi
echo "Port: $PORT"
if [ -n "$RESUME_PATH" ]; then
    echo "Resume: $RESUME_PATH"
fi
echo "============================================"

STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PATH_OUTPUT_XDECOMPOSER_ROOT}/${EXP_NAME}/${STAMP}"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME_ARGS=()
if [ -n "$RESUME_PATH" ]; then
    CLEAN_PATH="$RESUME_PATH"
    if [[ "$RESUME_PATH" == *".ptbash" ]]; then
        CLEAN_PATH="${RESUME_PATH%bash}"
    fi
    RESUME_ARGS=(--resume "$CLEAN_PATH")
    OUTPUT_DIR=$(dirname "$CLEAN_PATH")
fi

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    --rdzv_endpoint=localhost:$PORT \
    scripts/python_runners/train_xdecomposer.py \
    --experiment_name "$EXP_NAME" \
    --save_dir "$OUTPUT_DIR" \
    "${RESUME_ARGS[@]}" \
    --singlephase_xrd_db "$PATH_DATA_SINGLEPHASE" \
    --xrd_length 3500 \
    --num_phases 4 \
    --batch_size 128 \
    --num_workers "$NUM_WORKERS" \
    --augment \
    --mae_checkpoint "$PATH_CKPT_PRETRAIN" \
    --cnn_channels 48 96 192 384 \
    --cnn_kernels 15 8 8 10 \
    --cnn_strides 1 2 2 5 \
    --epochs 2000 \
    --lr 2e-4 \
    --lr_scheduler "cosine" \
    --warmup_epochs 20 \
    --noam_factor 1.0 \
    --alpha 5.0 \
    --lambda_sisdr 0.5 \
    --lambda_geo 5.0 \
    --beta 2 \
    --lambda_mix 5 \
    --lambda_activity 2.0 \
    --activity_threshold 0.8 \
    --vis_interval 50
