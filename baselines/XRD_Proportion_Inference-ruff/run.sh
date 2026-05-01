#!/bin/bash
#SBATCH --job-name=xrd_train
#SBATCH --partition=your-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

# Load the environment
module load miniconda
source activate xrd

# Set PYTHONPATH if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training
# Keep logs unbuffered
python -u Neural_network/train.py
