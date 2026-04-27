#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "$PROJECT_ROOT/configs/paths.sh"
require_conda_env
cd "$PROJECT_ROOT"
export LD_LIBRARY_PATH="${ENV_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.

STAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="${PATH_OUTPUT_MP20_ROOT}_${STAMP}"

for K in 2 3 4; do
    echo "====================================="
    echo "Starting MP20 test: k=${K}"
    echo "====================================="

    python scripts/python_runners/test_xdecomposer.py \
        --checkpoint "$PATH_CKPT_XDECOMPOSER" \
        --data_dir "$PATH_DATA_SINGLEPHASE" \
        --crystal_db "$PATH_DATA_CRYSTAL_DB" \
        --save_dir "${BASE_OUT}/k${K}" \
        --split test \
        --batch_size 128 \
        --min_k "$K" \
        --max_k "$K" \
        --k_weights 1.0 \
        --alpha 0.5 \
        --margin 5 \
        --hard_threshold 0.5
done

echo "All MP20 tests completed."
