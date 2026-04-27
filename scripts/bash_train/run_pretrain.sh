#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "$PROJECT_ROOT/configs/paths.sh"
require_conda_env
cd "$PROJECT_ROOT"

NUM_GPUS="$(detect_num_gpus)"
PORT=$((29500 + $RANDOM % 100))
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PATH_OUTPUT_PRETRAIN_ROOT}/${STAMP}"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME_ARGS=()
if [ -n "${1:-}" ]; then
    echo "Resuming from: $1"
    RESUME_ARGS=(--resume "$1")
    OUTPUT_DIR=$(dirname "$1")
fi
echo "Output Directory: $OUTPUT_DIR"
echo "Using GPUs: $NUM_GPUS"

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=localhost:$PORT \
    scripts/python_runners/train_pretrain.py \
    --experiment_name "mae_pretrain" \
    --output_dir "$OUTPUT_DIR" \
    "${RESUME_ARGS[@]}" \
    --singlephase_db "$PATH_DATA_SINGLEPHASE" \
    --xrd_length 3500 \
    --norm_method "max" \
    --batch_size 256 \
    --d_model 768 \
    --n_heads 12 \
    --n_layers 4 \
    --decoder_dim 512 \
    --decoder_heads 8 \
    --decoder_layers 4 \
    --patch_len 50 \
    --stride 25 \
    --dropout 0.1 \
    --epochs 2000 \
    --mask_ratio 0.70 \
    --ohem_ratio 0.5 \
    --lr 5e-4 \
    --lr_scheduler "cosine" \
    --warmup_epochs 50 \
    --noam_factor 1.0 \
    --alpha 10.0 \
    --lambda_cos 0.5 \
    --lambda_deriv 0.1
