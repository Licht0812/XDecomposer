#!/bin/bash
#SBATCH --job-name=xrd_train_gen
#SBATCH --partition=your-partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm_train_gen-%j.out

# Load the environment
module load miniconda
source activate xrd

# Force CPU execution
# export CUDA_VISIBLE_DEVICES="-1"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting training with generator at $(date)"
python -u train_with_generator.py
echo "Training finished at $(date)"
