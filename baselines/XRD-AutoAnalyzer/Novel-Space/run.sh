#!/bin/bash
#SBATCH --job-name=xrd_train
#SBATCH --partition=your-partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

module load miniconda
source activate xrd

# Disable GPU
export CUDA_VISIBLE_DEVICES=""

export PYTHONPATH=$PYTHONPATH:$(cd .. && pwd)
python -u construct_xrd_model.py --cif_dir=Cleaned_CIFs --num_spectra=1 --num_cpu=32 --skip_filter
