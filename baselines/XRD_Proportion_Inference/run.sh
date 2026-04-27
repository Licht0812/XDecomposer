#!/bin/bash
#SBATCH --job-name=xrd_train
#SBATCH --partition=project1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1             # 申请单卡 GPU

# 环境加载
module load miniconda
source activate xrd

# 设置 Python 路径（如果需要）
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 执行训练脚本
# -u 参数确保输出实时刷新到日志文件
python -u Neural_network/train.py
