#!/bin/bash
#SBATCH --job-name=xrd_eval_multi
#SBATCH --partition=project1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=slurm_eval_multi-%j.out

# 加载环境
module load miniconda
source activate xrd

# 强制使用 CPU，避免大规模分类时的显存溢出和 CUDA 冲突
# export CUDA_VISIBLE_DEVICES="-1"

# 设置 PYTHONPATH 确保能找到 autoXRD 模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting multi-phase evaluation at $(date)"

for num_phases in 2 3 4
do
    echo "Running evaluation for $num_phases phases..."
    python -u evaluate_multi_phase.py --num_phases $num_phases
done

echo "Evaluation finished at $(date)"
