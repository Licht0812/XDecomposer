#!/bin/bash
#SBATCH --job-name=xrd_train
#SBATCH --partition=project1      # 使用 CPU 队列或通用队列
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       # 增加 CPU 核心数以加速训练
#SBATCH --mem=256G

module load miniconda
source activate xrd

# 禁用 GPU，强制使用 CPU
export CUDA_VISIBLE_DEVICES=""

# rm -rf /data/home/zdhs0019/Projects/xrd_baselines/XRD-AutoAnalyzer/Novel-Spacev2/References
export PYTHONPATH=$PYTHONPATH:/data/home/zdhs0019/Projects/xrd_baselines/XRD-AutoAnalyzer
python -u construct_xrd_model.py --cif_dir=Cleaned_CIFs --num_spectra=1 --num_cpu=32 --skip_filter