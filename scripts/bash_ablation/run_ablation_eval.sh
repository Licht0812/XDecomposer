#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source configs/paths.sh
require_conda_env

export LD_LIBRARY_PATH="${ENV_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.

PYTHON_BIN=${PYTHON_BIN:-python}

RUN_MP20=${RUN_MP20:-1}
QUICK_MODE=${QUICK_MODE:-1}
REPEATS=${REPEATS:-1}
NUM_VIS=${NUM_VIS:-0}
START_SEED=${START_SEED:-42}
START_FROM=${START_FROM:-full}
OUT_DIR=${OUT_DIR:-"${PATH_OUTPUT_ABLATION_EVAL_ROOT}_$(date +%Y%m%d_%H%M%S)"}
FULL_CKPT=${FULL_CKPT:-}

MODELS=(
    full
    exp2_wo_transformer
    exp3_wo_film
    exp4_wo_geo_loss
    exp5_wo_skip_connections
    exp6_mask_direct
    exp7_mask_hard
)

TASKS=()
if [[ "$RUN_MP20" -eq 1 ]]; then
    TASKS+=("mp20:2" "mp20:3" "mp20:4")
fi

latest_checkpoint_under() {
    local root="$1"
    local best=""
    local latest=""

    best=$(find "$root" -type f -path "*/best.pt" 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n "$best" ]]; then
        echo "$best"
        return 0
    fi

    latest=$(find "$root" -type f -path "*/latest.pt" 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n "$latest" ]]; then
        echo "$latest"
        return 0
    fi

    return 1
}

resolve_ckpt() {
    local model_name="$1"

    if [[ "$model_name" == "full" ]]; then
        if [[ -n "$FULL_CKPT" && -f "$FULL_CKPT" ]]; then
            echo "$FULL_CKPT"
            return 0
        fi
        if [[ -f "$PATH_CKPT_XDECOMPOSER" ]]; then
            echo "$PATH_CKPT_XDECOMPOSER"
            return 0
        fi
        latest_checkpoint_under "$PATH_OUTPUT_XDECOMPOSER_ROOT"
        return $?
    fi

    latest_checkpoint_under "$PATH_OUTPUT_ABLATION_ROOT/${model_name}"
}

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Ablation evaluation"
echo "Output     : $OUT_DIR"
echo "Start from : $START_FROM"
echo "Tasks      : ${TASKS[*]}"
echo "Quick mode : $QUICK_MODE"
echo "Repeats    : $REPEATS"
echo "=============================================="

START_REACHED=0
FAILED=0

for MODEL_NAME in "${MODELS[@]}"; do

    if [[ "$START_REACHED" -eq 0 ]]; then
        if [[ "$MODEL_NAME" != "$START_FROM" ]]; then
            echo "Skipping $MODEL_NAME"
            continue
        fi
        START_REACHED=1
    fi

    if ! CKPT="$(resolve_ckpt "$MODEL_NAME")"; then
        echo "[SKIP] $MODEL_NAME checkpoint not found"
        FAILED=$((FAILED + 1))
        continue
    fi

    for task in "${TASKS[@]}"; do
        DATASET="${task%%:*}"
        K="${task#*:}"

        for ((r=0; r<REPEATS; r++)); do
            SEED=$((START_SEED + r))
            SAVE_DIR="$OUT_DIR/$MODEL_NAME/${DATASET}_k${K}/seed_${SEED}"
            mkdir -p "$SAVE_DIR"

            CMD=(
                "$PYTHON_BIN" scripts/python_runners/test_xdecomposer.py
                --checkpoint "$CKPT"
                --save_dir "$SAVE_DIR"
                --split test
                --seed "$SEED"
                --num_vis "$NUM_VIS"
                --min_k "$K"
                --max_k "$K"
                --k_weights 1.0
                --alpha 0.5
                --margin 5
                --hard_threshold 0.5
            )

            if [[ "$QUICK_MODE" -eq 1 ]]; then
                CMD+=(--quick)
            fi

            CMD+=(
                --data_dir "$PATH_DATA_SINGLEPHASE"
                --crystal_db "$PATH_DATA_CRYSTAL_DB"
                --batch_size 128
            )

            echo "[RUN] model=$MODEL_NAME dataset=$DATASET k=$K seed=$SEED"

            if ! "${CMD[@]}" 2>&1 | tee "$SAVE_DIR/runner.log"; then
                echo "[FAIL] model=$MODEL_NAME dataset=$DATASET k=$K seed=$SEED"
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

if [[ "$START_REACHED" -eq 0 ]]; then
    echo "ERROR: START_FROM=$START_FROM not found"
    exit 1
fi

echo "Generating summary CSV..."

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv
import glob
import json
import os
import statistics
import sys

out_dir = sys.argv[1]
files = glob.glob(os.path.join(out_dir, "*", "*", "seed_*", "test_metrics.json"))

if not files:
    print("No metrics files found.")
    sys.exit(0)

rows = []
for p in sorted(files):
    rel = os.path.relpath(p, out_dir)
    model, task, seed_dir, _ = rel.split(os.sep)[:4]

    with open(p, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    row = {
        "model": model,
        "task": task,
        "seed": seed_dir.replace("seed_", ""),
    }
    row.update(metrics)
    rows.append(row)

metric_keys = sorted({k for r in rows for k in r if k not in {"model", "task", "seed"}})

raw_csv = os.path.join(out_dir, "all_runs_metrics.csv")
with open(raw_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "task", "seed"] + metric_keys)
    writer.writeheader()
    writer.writerows(rows)

focus = [
    "loss",
    "si_sdr",
    "pearson_corr",
    "sir",
    "sar",
    "delta_2theta",
    "fwhm_error",
    "id_acc_top1",
    "id_acc_top10",
]

groups = {}
for r in rows:
    groups.setdefault((r["model"], r["task"]), []).append(r)

summary = []
for (model, task), group in sorted(groups.items()):
    out = {"model": model, "task": task, "n": len(group)}

    for m in focus:
        vals = [float(r[m]) for r in group if m in r and r[m] != ""]
        out[f"{m}_mean"] = statistics.mean(vals) if vals else ""
        out[f"{m}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else ""

    summary.append(out)

summary_csv = os.path.join(out_dir, "summary_focus_metrics.csv")
fields = ["model", "task", "n"]
for m in focus:
    fields += [f"{m}_mean", f"{m}_std"]

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print(f"Saved: {raw_csv}")
print(f"Saved: {summary_csv}")
PY

if [[ "$FAILED" -gt 0 ]]; then
    echo "Completed with failures: $FAILED"
    echo "Output: $OUT_DIR"
    exit 2
fi

echo "All evaluations completed successfully."
echo "Output: $OUT_DIR"
