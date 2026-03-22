#!/bin/bash
# ==========================================

source "$(dirname "$0")/../../configs/paths.sh"
source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
cd "$(dirname "$0")/../.."

SAVE_DIR="${PATH_OUTPUT_TEST}/default_eval_$(date +%Y%m%d_%H%M%S)"

echo "====================================="
echo "🎯 Starting default evaluation pipeline"
echo "Checkpoint: $PATH_CKPT_SEP"
echo "====================================="

python scripts/python_runners/test_separation_film.py \
    --checkpoint "$PATH_CKPT_SEP" \
    --data_dir "$PATH_DATA_SINGLEPHASE" \
    --crystal_db "$PATH_DATA_CRYSTAL_DB" \
    --save_dir "$SAVE_DIR" \
    --split test \
    --batch_size 128 \
    --num_vis 20

echo "Evaluation complete! Logs and metrics saved in $SAVE_DIR."
