#!/bin/bash
# Import all unified variable paths
source "$(dirname "$0")/../../configs/paths.sh"

# Activate environment and resolve PyTorch and CUDA dynamic library errors
source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
export LD_LIBRARY_PATH=$ENV_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=. 

echo "====================================="
echo "🎯 1/3 Starting RRUFF data test: Pure 2-phase mixture (k=2)"
echo "====================================="
python scripts/python_runners/test_separation_film.py \
    --checkpoint $PATH_CKPT_SEP_RRUFF \
    --data_dir $PATH_DATA_RRUFF \
    --save_dir "${PATH_OUTPUT_TEST}/film_rruff_k2_only" \
    --batch_size 32 \
    --min_k 2 \
    --max_k 2 \
    --k_weights 1.0 \
    --alpha 0.5 \
    --margin 5 \
    --hard_threshold 0.5 \
    --num_vis 20

echo "====================================="
echo "🎯 2/3 Starting RRUFF data test: 2-phase & 3-phase mixture (k=3, ratio 0.33 0.67)"
echo "====================================="
python scripts/python_runners/test_separation_film.py \
    --checkpoint $PATH_CKPT_SEP_RRUFF \
    --data_dir $PATH_DATA_RRUFF \
    --save_dir "${PATH_OUTPUT_TEST}/film_rruff_k3_mixed" \
    --batch_size 32 \
    --min_k 2 \
    --max_k 3 \
    --k_weights 0.67 0.33 \
    --alpha 0.5 \
    --margin 5 \
    --hard_threshold 0.5 \
    --num_vis 20
    
echo "====================================="
echo "🎯 3/3 Starting RRUFF data test: 2-phase, 3-phase & 4-phase mixture (k=4, ratio 0.6 0.25 0.15)"
echo "====================================="
python scripts/python_runners/test_separation_film.py \
    --checkpoint $PATH_CKPT_SEP_RRUFF \
    --data_dir $PATH_DATA_RRUFF \
    --save_dir "${PATH_OUTPUT_TEST}/film_rruff_k4_mixed" \
    --batch_size 32 \
    --min_k 2 \
    --max_k 4 \
    --k_weights 0.6 0.25 0.15 \
    --alpha 0.5 \
    --margin 5 \
    --hard_threshold 0.5 \
    --num_vis 20

echo "🎉 All three RRUFF tests have been successfully executed!"
