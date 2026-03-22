#!/bin/bash
# ==========================================
# 1. Environment Configuration
export CONDA_ACTIVATE_PATH="$HOME/anaconda3/bin/activate"
export CONDA_ENV_NAME="spectra"
export ENV_LD_LIBRARY_PATH="$HOME/.conda/envs/spectra/lib/python3.10/site-packages/nvidia/nvjitlink/lib"

# 2. Dataset Paths 
export PATH_DATA_SINGLEPHASE="/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data"
export PATH_DATA_CRYSTAL_DB="/data/group/project1/Crystal/UniqCryLabeled.db"
export PATH_DATA_RRUFF="XRDUnmix/data/UniqRruffCrystal.db"

# 3. Model and Checkpoint Paths
export PATH_CKPT_MAE="checkpoints/pretrain/checkpoint_latest.pt"
export PATH_CKPT_SEP="checkpoints/separation/latest.pt"
# 4. Output Logs and Test Results Directory
export PATH_OUTPUT_TEST="test_results"
