#!/bin/bash
#SBATCH -c 12

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

source configs/paths.sh
require_conda_env

export PYTHONPATH=.
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-XDecomposer-Ablation}"

PYTHON_BIN=${PYTHON_BIN:-python}
SAVE_DIR=${SAVE_DIR:-$PATH_OUTPUT_ABLATION_ROOT}
CONFIG_DIR=${CONFIG_DIR:-$PATH_DIR_ABLATION_CONFIGS}
START_FROM=${START_FROM:-exp2_wo_transformer}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
NUM_WORKERS=${NUM_WORKERS:-0}

NUM_GPUS="$(detect_num_gpus)"

mkdir -p "$SAVE_DIR"

EXPERIMENTS=(
    "exp2_wo_transformer:$CONFIG_DIR/exp2_wo_transformer.yaml"
    "exp3_wo_film:$CONFIG_DIR/exp3_wo_film.yaml"
    "exp4_wo_geo_loss:$CONFIG_DIR/exp4_wo_geo_loss.yaml"
    "exp5_wo_skip_connections:$CONFIG_DIR/exp5_wo_skip_connections.yaml"
    "exp6_mask_direct:$CONFIG_DIR/exp6_mask_direct.yaml"
    "exp7_mask_hard:$CONFIG_DIR/exp7_mask_hard.yaml"
)

COMMON_ARGS=(
    --mae_checkpoint "$PATH_CKPT_PRETRAIN"
    --singlephase_xrd_db "$PATH_DATA_SINGLEPHASE"
    --xrd_length 3500
    --num_phases 4
    --batch_size 128
    --num_workers "$NUM_WORKERS"
    --augment
    --cnn_channels 48 96 192 384
    --cnn_kernels 15 8 8 10
    --cnn_strides 1 2 2 5
    --epochs 500
    --lr 2e-4
    --lr_scheduler cosine
    --warmup_epochs 20
    --noam_factor 1.0
    --alpha 5.0
    --lambda_geo 5.0
    --lambda_sisdr 0.5
    --beta 2
    --lambda_mix 5
    --lambda_activity 2.0
    --activity_threshold 0.8
)

START_REACHED=0

for exp in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${exp%%:*}"
    CONFIG_FILE="${exp#*:}"

    if [[ "$START_REACHED" -eq 0 ]]; then
        if [[ "$EXP_NAME" != "$START_FROM" ]]; then
            echo "Skipping $EXP_NAME"
            continue
        fi
        START_REACHED=1
    fi

    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: config not found: $CONFIG_FILE"
        exit 1
    fi

    EXP_DIR="$SAVE_DIR/$EXP_NAME/$RUN_STAMP"
    LOG_FILE="$EXP_DIR/launcher.log"
    SWAN_EXP_NAME="$EXP_NAME"

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        SWAN_EXP_NAME="${EXP_NAME}_job${SLURM_JOB_ID}"
    fi

    mkdir -p "$EXP_DIR"

    echo "=========================================="
    echo "Training $EXP_NAME"
    echo "Config    : $CONFIG_FILE"
    echo "Save dir  : $EXP_DIR"
    echo "GPUs      : $NUM_GPUS"
    echo "Swan name : $SWAN_EXP_NAME"
    echo "=========================================="

    OMP_NUM_THREADS=4 "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" \
        scripts/python_runners/train_xdecomposer.py \
        "${COMMON_ARGS[@]}" \
        --experiment_name "$SWAN_EXP_NAME" \
        --save_dir "$EXP_DIR" \
        --config "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"
done

if [[ "$START_REACHED" -eq 0 ]]; then
    echo "ERROR: START_FROM=$START_FROM not found"
    exit 1
fi

echo "All ablation trainings completed."
