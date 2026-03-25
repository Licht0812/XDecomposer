#!/bin/bash
#SBATCH --job-name=xrd_infer
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

# 设置模型加载路径，用户可以根据需要修改
LOAD_PATH="/data/home/zdhs0019/Projects/xrd_baselines/XQueryer/output/2026-03-24_1725/checkpoints/checkpoint_0010.pth"

# 执行推理脚本
# -u 参数确保输出实时刷新到日志文件
# limit 0 表示评估全部数据，或者设置一个小数值（如 100）进行快速测试
python -u src/infer.py \
    --load_path "$LOAD_PATH" \
    --batch_size 8 \
    --limit 0 \
    --threshold 0.3 \
    --num_slots 4 \
    --feature_dim 256
