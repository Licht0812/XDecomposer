#!/bin/bash
source "$(dirname "$0")/../../configs/paths.sh"

source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
export LD_LIBRARY_PATH=$ENV_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=. 

echo "====================================="
echo "1/3 Starting test: Pure 2-phase mixture (k=2)"
echo "====================================="
python scripts/python_runners/test_separation_film.py \
    --checkpoint $PATH_CKPT_SEP \
    --data_dir $PATH_DATA_SINGLEPHASE \
    --crystal_db $PATH_DATA_CRYSTAL_DB \
    --save_dir "${PATH_OUTPUT_TEST}/film_mp_k2_only" \
    --split test \
    --batch_size 128 \
    --min_k 2 \
    --max_k 2 \
    --k_weights 1.0 \
    --alpha 0.5 \
    --margin 5 \
    --hard_threshold 0.5

echo "====================================="
echo "2/3 Starting test: 2-phase & 3-phase mixture (k=3, ratio 0.67 0.33)"
echo "====================================="
python scripts/python_runners/test_separation_film.py \
    --checkpoint $PATH_CKPT_SEP \
    --data_dir $PATH_DATA_SINGLEPHASE \
    --crystal_db $PATH_DATA_CRYSTAL_DB \
    --save_dir "${PATH_OUTPUT_TEST}/film_mp_k3_mixed" \
    --split test \
    --batch_size 128 \
    --min_k 2 \
    --max_k 3 \
    --k_weights 0.67 0.33 \
    --alpha 0.5 \
    --margin 5 \
    --hard_threshold 0.5
    
echo "====================================="
echo "3/3 Starting test: 2-phase, 3-phase & 4-phase mixture (k=4, ratio 0.6 0.25 0.15)"
echo "====================================="
python scripts/python_runners/test_separation_film.py \
    --checkpoint $PATH_CKPT_SEP \
    --data_dir $PATH_DATA_SINGLEPHASE \
    --crystal_db $PATH_DATA_CRYSTAL_DB \
    --save_dir "${PATH_OUTPUT_TEST}/film_mp_k4_mixed" \
    --split test \
    --batch_size 128 \
    --min_k 2 \
    --max_k 4 \
    --k_weights 0.6 0.25 0.15 \
    --alpha 0.5 \
    --margin 5 \
    --hard_threshold 0.5

echo "All three tests have been successfully executed!"
