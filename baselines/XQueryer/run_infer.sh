#!/bin/bash
#SBATCH --job-name=xrd_infer
#SBATCH --partition=your-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

# Load the environment
module load miniconda
source activate xrd

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set the checkpoint path
LOAD_PATH="checkpoints/xqueryer/latest.pth"

# Keep logs unbuffered
# Use 0 to evaluate the full set

echo "Starting inference at $(date)"

for num_phases in 2 3 4
do
    echo "Running inference for $num_phases phases..."
    python -u src/infer.py \
        --load_path "$LOAD_PATH" \
        --batch_size 8 \
        --limit 0 \
        --threshold 0.3 \
        --num_slots 4 \
        --feature_dim 256 \
        --num_phases $num_phases
done

echo "Inference finished at $(date)"
