#!/bin/bash
#SBATCH --job-name=xrd_eval
#SBATCH --partition=project1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1             # 评估通常也使用 GPU 加速

# 环境加载
module load miniconda
source activate xrd

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 执行测试脚本
# 默认会读取当前目录下的 best_proportion_model.pth
python -u Neural_network/test.py
