#!/bin/bash
#SBATCH --job-name=xrd_kfold_train
#SBATCH --partition=project1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

# 环境加载
module load miniconda
source activate xrd

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

DB_PATH="/data/group/project1/Crystal/UniqRruffCrystal.db"
NUM_FOLDS=5
EPOCHS=100

echo "🚀 Starting ${NUM_FOLDS}-Fold Cross-Validation Training on RRUFF Database..."

for fold in $(seq 0 $((NUM_FOLDS-1))); do
    echo "==========================================="
    echo "🔥 Starting Fold $((fold+1)) / ${NUM_FOLDS}"
    echo "==========================================="

    python -u src/train.py \
        --db_path $DB_PATH \
        --batch_size 32 \
        --epochs $EPOCHS \
        --lr 8e-5 \
        --num_classes 100315 \
        --num_slots 4 \
        --feature_dim 256 \
        --atom_embed True \
        --patience 10 \
        --progress_bar True \
        --num_folds $NUM_FOLDS \
        --fold $fold

    echo "✅ Fold $((fold+1)) Complete."
done

echo "🎉 All folds training completed."
