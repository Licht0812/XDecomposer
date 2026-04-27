#!/bin/bash

# Runtime
export CONDA_ENV_NAME="xdecomposer"

require_conda_env() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV_NAME" ]; then
        return 0
    fi
    if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "$CONDA_ENV_NAME" ]; then
        return 0
    fi
    echo "Activate conda env '$CONDA_ENV_NAME' before running this script." >&2
    exit 1
}

count_csv_items() {
    local value="$1"
    if [ -z "$value" ]; then
        echo 0
        return
    fi
    IFS=',' read -ra ITEMS <<< "$value"
    echo "${#ITEMS[@]}"
}

detect_num_gpus() {
    local value="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
    local count=""

    if [ -n "$value" ]; then
        if [[ "$value" == *"("* ]]; then
            value="${value%%(*}"
        fi
        if [[ "$value" == *":"* ]]; then
            value="${value##*:}"
        fi
        if [[ "$value" =~ ^[0-9]+$ ]] && [ "$value" -gt 0 ]; then
            echo "$value"
            return
        fi
    fi

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        count=$(count_csv_items "$CUDA_VISIBLE_DEVICES")
        if [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ]; then
            echo "$count"
            return
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ]; then
            echo "$count"
            return
        fi
    fi

    echo 1
}

# Project paths
export PATH_DIR_CHECKPOINTS="checkpoints"
export PATH_DIR_PRETRAIN="$PATH_DIR_CHECKPOINTS/pretrain"
export PATH_DIR_PRETRAIN_RUNS="$PATH_DIR_PRETRAIN/mae_pretrain"
export PATH_DIR_XDECOMPOSER="$PATH_DIR_CHECKPOINTS/xdecomposer"
export PATH_DIR_RRUFF_FINETUNE="$PATH_DIR_CHECKPOINTS/rruff_finetune"
export PATH_DIR_ABLATION="$PATH_DIR_CHECKPOINTS/ablation"
export PATH_DIR_TEST_RESULTS="test_results"
export PATH_DIR_ABLATION_CONFIGS="configs/ablation_configs"
export PATH_FILE_BASELINE_TRAIN_CONFIG="baselines/transformer_family/train_config.yaml"

# Data
export PATH_DATA_SINGLEPHASE="mp20-xrd_data/data"
export PATH_DATA_CRYSTAL_DB="data/UniqCryLabeled.db"
export PATH_DATA_RRUFF="data/UniqRruffCrystal.db"

# Checkpoints
export PATH_CKPT_PRETRAIN="$PATH_DIR_PRETRAIN/checkpoint_latest.pt"
export PATH_CKPT_XDECOMPOSER_MP20="$PATH_DIR_XDECOMPOSER/latest.pt"
export PATH_CKPT_XDECOMPOSER="$PATH_CKPT_XDECOMPOSER_MP20"
export PATH_CKPT_XDECOMPOSER_BEST="$PATH_DIR_XDECOMPOSER/best.pt"
export PATH_CKPT_BASELINE="${PATH_CKPT_BASELINE:-}"

# Outputs
export PATH_OUTPUT_PRETRAIN_ROOT="$PATH_DIR_PRETRAIN_RUNS"
export PATH_OUTPUT_XDECOMPOSER_ROOT="$PATH_DIR_XDECOMPOSER"
export PATH_OUTPUT_ABLATION_ROOT="$PATH_DIR_ABLATION"
export PATH_OUTPUT_TEST="$PATH_DIR_TEST_RESULTS"
export PATH_OUTPUT_MP20_ROOT="$PATH_OUTPUT_TEST/mp20_k"
export PATH_OUTPUT_RRUFF_KFOLD_ROOT="$PATH_OUTPUT_TEST/rruff_kfold"
export PATH_OUTPUT_RRUFF_FINETUNE_ROOT="$PATH_DIR_RRUFF_FINETUNE"
export PATH_OUTPUT_ABLATION_EVAL_ROOT="$PATH_OUTPUT_TEST/ablation_eval"
export PATH_OUTPUT_XDECOMPOSER_EVAL="$PATH_OUTPUT_TEST/xdecomposer_eval"
export PATH_OUTPUT_BASELINE_ROOT="$PATH_OUTPUT_TEST/transformer_family"
export PATH_OUTPUT_BASELINE_EVAL="$PATH_OUTPUT_TEST/transformer"
export PATH_OUTPUT_BASELINE_RRUFF_EVAL="$PATH_OUTPUT_TEST/transformer_family_rruff_kfold"
export PATH_TEMPLATE_BASELINE_SAVE_DIR="$PATH_DIR_CHECKPOINTS/transformer_{baseline_name}_{run_name}_{timestamp}"
