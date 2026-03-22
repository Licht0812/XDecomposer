#!/bin/bash
# ==========================================
# RRUFF Fine-Tuning 5-Fold Cross Validation Script
# ==========================================

source "$(dirname "$0")/../../configs/paths.sh"

source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
export LD_LIBRARY_PATH=$ENV_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=. 

NUM_FOLDS=5
GPUS="0,1"  # Adjusted appropriately

# Provide the best pre-trained checkpoint from MP20 here:
BASE_CKPT=$PATH_CKPT_SEP
echo "Starting RRUFF 5-Fold Fine-tuning using Base MP20 Model: $BASE_CKPT"

for fold in $(seq 0 $((NUM_FOLDS-1))); do
    echo "=========================================================="
    echo "🚀 Starting Fold ${fold}/${NUM_FOLDS}..."
    echo "=========================================================="
    
    EXP_NAME="rruff_finetune_fold_${fold}"
    OUTPUT_DIR="checkpoints/${EXP_NAME}"
    PORT=$((29500 + fold))

    # 1. Fine-tune on RRUFF training split
    echo "[Train] Fine-tuning on Fold ${fold}..."
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=$PORT \
        scripts/python_runners/finetune_rruff.py \
        --experiment_name $EXP_NAME \
        --save_dir $OUTPUT_DIR \
        --singlephase_xrd_db $PATH_DATA_RRUFF \
        --mae_checkpoint $PATH_CKPT_MAE \
        --finetune $BASE_CKPT \
        --batch_size 64 \
        --num_phases 4 \
        --cnn_channels 48 96 192 384 \
        --epochs 100 \
        --lr 5e-5 \
        --fold $fold \
        --num_folds $NUM_FOLDS \
        --seed 42

    # 2. Test on RRUFF testing split
    echo "[Test] Evaluating on Fold ${fold} test subset..."
    python scripts/python_runners/test_separation_film.py \
        --checkpoint "${OUTPUT_DIR}/best.pt" \
        --data_dir $PATH_DATA_RRUFF \
        --save_dir "${PATH_OUTPUT_TEST}/rruff_kfold_results/fold_${fold}" \
        --batch_size 32 \
        --split test \
        --fold $fold \
        --num_folds $NUM_FOLDS \
        --seed 42 \
        --alpha 0.5 \
        --margin 5 \
        --hard_threshold 0.5
        
    echo "Fold ${fold} Complete."
done

echo "🎉 All Folds Complete!"

echo "📊 Aggregating final metrics across all folds..."
python scripts/bash_train/aggregate_metrics.py
