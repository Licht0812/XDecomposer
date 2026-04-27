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
    echo "Usage: baselines/transformer_family/eval_mp20_k.sh --checkpoint path/to/best.pt [--baseline transformer|itransformer|patchtst]"
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
OUT_ROOT="${PATH_OUTPUT_BASELINE_ROOT}_${CKPT_NAME}_${STAMP}"

echo "====================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output root: $OUT_ROOT"
echo "====================================="

echo "====================================="
echo "1/3 Starting test: Pure 2-phase mixture (k=2)"
echo "====================================="
python baselines/transformer_family/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    "${BASELINE_ARG[@]}" \
    --mae_checkpoint "$PATH_CKPT_PRETRAIN" \
    --data_dir "$PATH_DATA_SINGLEPHASE" \
    --crystal_db "$PATH_DATA_CRYSTAL_DB" \
    --save_dir "${OUT_ROOT}_k2" \
    --split test \
    --batch_size 128 \
    --min_k 2 \
    --max_k 2 \
    --k_weights 1.0 \
    --activity_threshold 0.5 \
    "${EXTRA_ARGS[@]}"

echo "====================================="
echo "2/3 Starting test: Pure 3-phase mixture (k=3)"
echo "====================================="
python baselines/transformer_family/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    "${BASELINE_ARG[@]}" \
    --mae_checkpoint "$PATH_CKPT_PRETRAIN" \
    --data_dir "$PATH_DATA_SINGLEPHASE" \
    --crystal_db "$PATH_DATA_CRYSTAL_DB" \
    --save_dir "${OUT_ROOT}_k3" \
    --split test \
    --batch_size 128 \
    --min_k 3 \
    --max_k 3 \
    --k_weights 1.0 \
    --activity_threshold 0.5 \
    "${EXTRA_ARGS[@]}"

echo "====================================="
echo "3/3 Starting test: Pure 4-phase mixture (k=4)"
echo "====================================="
python baselines/transformer_family/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    "${BASELINE_ARG[@]}" \
    --mae_checkpoint "$PATH_CKPT_PRETRAIN" \
    --data_dir "$PATH_DATA_SINGLEPHASE" \
    --crystal_db "$PATH_DATA_CRYSTAL_DB" \
    --save_dir "${OUT_ROOT}_k4" \
    --split test \
    --batch_size 128 \
    --min_k 4 \
    --max_k 4 \
    --k_weights 1.0 \
    --activity_threshold 0.5 \
    "${EXTRA_ARGS[@]}"

echo "All three baseline tests have been successfully executed."
