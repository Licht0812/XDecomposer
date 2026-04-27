#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "$PROJECT_ROOT/configs/paths.sh"
require_conda_env
cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="${ENV_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.

NUM_FOLDS=5
CKPT="$PATH_CKPT_XDECOMPOSER"
STAMP=$(date +%Y%m%d_%H%M%S)

for K in 2 3 4; do
    echo "====================================="
    echo "Starting RRUFF fold test: k=${K}"
    echo "====================================="

    BASE_OUT="${PATH_OUTPUT_RRUFF_KFOLD_ROOT}_k${K}_${STAMP}"
    mkdir -p "$BASE_OUT"

    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        echo "Evaluating k=${K}, fold=${fold}, checkpoint=${CKPT}"

        python scripts/python_runners/test_xdecomposer.py \
            --checkpoint "$CKPT" \
            --data_dir "$PATH_DATA_RRUFF" \
            --save_dir "${BASE_OUT}/fold_${fold}" \
            --split test \
            --batch_size 32 \
            --min_k "$K" \
            --max_k "$K" \
            --alpha 0.5 \
            --margin 5 \
            --hard_threshold 0.5 \
            --fold "$fold" \
            --num_folds "$NUM_FOLDS" \
            --seed 42 \
            --num_vis 2
    done

    python scripts/bash_train/aggregate_metrics.py "$BASE_OUT"
done

echo "All RRUFF fold tests completed."
