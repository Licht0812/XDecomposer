#!/bin/bash
#SBATCH --job-name=xrd_kfold_eval
#SBATCH --partition=project1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

module load miniconda
source activate xrd
export PYTHONPATH=$PYTHONPATH:$(pwd)

DB_PATH="/data/group/project1/Crystal/UniqRruffCrystal.db"
NUM_FOLDS=5
NUM_CLASSES=100315

# Modify this array to match the output paths from your training run
# Example: 
# CHECKPOINTS=(
#     "output/fold_0/model_best.pth"
#     "output/fold_1/model_best.pth"
#     "output/fold_2/model_best.pth"
#     "output/fold_3/model_best.pth"
#     "output/fold_4/model_best.pth"
# )

# Assuming we have a way to find the latest run per fold:
# You'll need to manually set the CHECKPOINTS array or use a script to find them.

# Placeholder logic:
echo "Please update CHECKPOINTS array in this script before running if needed."

CHECKPOINTS=()
for fold in $(seq 0 $((NUM_FOLDS-1))); do
    # This finds the latest output directory for the fold
    LATEST_DIR=$(ls -td output/*_fold_${fold} | head -1)
    if [ -n "$LATEST_DIR" ]; then
        if [ -f "${LATEST_DIR}/checkpoints/model_best.pth" ]; then
            CHECKPOINTS+=("${LATEST_DIR}/checkpoints/model_best.pth")
        else
            LATEST_CHECKPOINT=$(ls -v ${LATEST_DIR}/checkpoints/checkpoint_*.pth 2>/dev/null | tail -1)
            if [ -n "$LATEST_CHECKPOINT" ]; then
                CHECKPOINTS+=("$LATEST_CHECKPOINT")
            else
                echo "No checkpoint found for fold ${fold}"
                exit 1
            fi
        fi
    else
        echo "No output directory found for fold ${fold}"
        exit 1
    fi
done

echo "Found checkpoints:"
printf '%s\n' "${CHECKPOINTS[@]}"

for phase in 2 3 4; do
    echo "==========================================="
    echo "🧪 Evaluating for ${phase} phases"
    echo "==========================================="
    
    for fold in $(seq 0 $((NUM_FOLDS-1))); do
        echo "Evaluating Fold ${fold}..."
        python -u src/infer.py \
            --db_path $DB_PATH \
            --batch_size 16 \
            --num_classes $NUM_CLASSES \
            --num_slots 4 \
            --feature_dim 256 \
            --atom_embed True \
            --load_path "${CHECKPOINTS[$fold]}" \
            --num_phases $phase \
            --num_folds $NUM_FOLDS \
            --fold $fold
    done
done

echo "🎉 Evaluation completed. You can now aggregate the JSON results for 2, 3, and 4 phases."
