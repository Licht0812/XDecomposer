#!/bin/bash
#SBATCH --job-name=xrd_train
#SBATCH --partition=your-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# Load the environment
module load miniconda
source activate xrd

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training
# Keep logs unbuffered
# Peak shift usually needs longer training
python -u src/train.py \
    --db_path data/UniqCryLabeled.db \
    --npz_dir data/UniqCry \
    --batch_size 32 \
    --epochs 200 \
    --lr 8e-5 \
    --num_classes 100315 \
    --num_slots 4 \
    --feature_dim 256 \
    --atom_embed True \
    --patience 10 \
    --progress_bar True
