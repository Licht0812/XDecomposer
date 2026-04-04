#!/bin/bash
#SBATCH --job-name=xrd_train_gen
#SBATCH --partition=project1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm_train_gen-%j.out

# 加载环境
# module load miniconda
# source activate xrd

# 强制使用 CPU，避免大规模分类时的显存溢出和 CUDA 冲突
# export CUDA_VISIBLE_DEVICES="-1"

# 设置 PYTHONPATH 确保能找到 autoXRD 模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting training with generator at $(date)"
python -u train_with_generator.py
echo "Training finished at $(date)"
