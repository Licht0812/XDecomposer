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

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 执行训练脚本
# -u 参数确保输出实时刷新到日志文件
# 启用了 Peak Shift 增强后，建议至少训练 100-200 个 Epoch
python -u src/train.py \
    --db_path /data/group/project1/Crystal/UniqCryLabeled.db \
    --npz_dir /data/group/project1/Crystal/UniqCry \
    --batch_size 32 \
    --epochs 200 \
    --lr 8e-5 \
    --num_classes 100315 \
    --num_slots 4 \
    --feature_dim 256 \
    --atom_embed True \
    --patience 10 \
    --progress_bar True
