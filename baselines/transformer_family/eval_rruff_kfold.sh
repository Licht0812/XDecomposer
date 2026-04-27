#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/configs/paths.sh"
source "$SCRIPT_DIR/env_utils.sh"

require_option_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == -* ]]; then
        echo "Missing value for $option" >&2
        exit 1
    fi
}

CHECKPOINT="${PATH_CKPT_BASELINE:-}"
BASELINE_ARG=()
GPUS="${CUDA_VISIBLE_DEVICES:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            require_option_value "$1" "${2:-}"
            CHECKPOINT="$2"
            shift 2
            ;;
        --baseline|--model)
            require_option_value "$1" "${2:-}"
            BASELINE_ARG=(--baseline_name "$2")
            shift 2
            ;;
        --gpus)
            require_option_value "$1" "${2:-}"
            GPUS="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            if [[ "$1" == -* ]]; then
                EXTRA_ARGS+=("$1")
                shift
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    EXTRA_ARGS+=("$1")
                    shift
                done
                continue
            elif [ -z "$CHECKPOINT" ]; then
                CHECKPOINT="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash baselines/transformer_family/eval_rruff_kfold.sh --checkpoint path/to/best.pt [--baseline transformer|itransformer|patchtst]"
    exit 1
fi

activate_spectra_env
cd "$PROJECT_ROOT"

if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi
export LD_LIBRARY_PATH="${ENV_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

CKPT_NAME=$(basename "$(dirname "$CHECKPOINT")")
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${PATH_OUTPUT_BASELINE_RRUFF_EVAL}_${CKPT_NAME}_${STAMP}"

echo "====================================="
echo "Checkpoint: $CHECKPOINT"
echo "RRUFF DB: $PATH_DATA_RRUFF"
echo "Output root: $OUT_ROOT"
echo "====================================="

python baselines/transformer_family/evaluate_rruff_kfold.py \
    --checkpoint "$CHECKPOINT" \
    "${BASELINE_ARG[@]}" \
    --mae_checkpoint "$PATH_CKPT_PRETRAIN" \
    --rruff_db "$PATH_DATA_RRUFF" \
    --save_dir "$OUT_ROOT" \
    --batch_size 128 \
    --num_folds 5 \
    --k_values 2 3 4 \
    --activity_threshold 0.5 \
    "${EXTRA_ARGS[@]}"

echo "RRUFF k-fold baseline evaluation completed."
