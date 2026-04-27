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
    BASE_OUT="${PATH_OUTPUT_RRUFF_FINETUNE_ROOT}_k${K}_${STAMP}"
    mkdir -p "$BASE_OUT"

    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        echo "=== K=${K}, Fold ${fold}/${NUM_FOLDS} ==="
        python scripts/python_runners/finetune_rruff.py \
            --checkpoint "$CKPT" \
            --data_dir "$PATH_DATA_RRUFF" \
            --save_dir "${BASE_OUT}/fold_${fold}" \
            --fold "$fold" --num_folds "$NUM_FOLDS" \
            --min_k "$K" --max_k "$K" \
            --seed 42
    done

    python scripts/bash_train/aggregate_metrics.py "$BASE_OUT"
done

echo "All RRUFF fine-tuning folds completed."
