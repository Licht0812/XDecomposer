#!/bin/bash
set -e

# ==========================================
# MAE Pre-training
# bash scripts/bash_train/run_pretrain.sh [resume_checkpoint_path]
# ==========================================

source "$(dirname "$0")/../../configs/paths.sh"
source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
cd "$(dirname "$0")/../.."

NUM_GPUS=8
PORT=$((29500 + $RANDOM % 100))
OUTPUT_DIR="checkpoints/mae_pretrain_$(date +%Y%m%d_%H%M%S)"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME_ARG=""
if [ -n "$1" ]; then
    echo "Resuming from: $1"
    RESUME_ARG="--resume $1"
    OUTPUT_DIR=$(dirname "$1")
fi
echo "Output Directory: $OUTPUT_DIR"

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=localhost:$PORT \
    scripts/python_runners/train_pretrain.py \
    --experiment_name "mae_pretrain" \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_ARG \
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
