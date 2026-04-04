#!/bin/bash

# run_eval.sh: Automates DARA evaluation with CPU optimization.

# --- Configuration ---
DB_PATH="/data/group/project1/Crystal/UniqCryLabeled.db"
# Updated to precise standard data directory
NPZ_DIR="/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data/"
CIF_DIR="/data/home/zdhs0019/Projects/xrd_baselines/dara/dataset/uniqcry_cifs"
OUTPUT_DIR="./evaluation_results-300"
MAX_SAMPLES=300 # Set to -1 for evaluating ALL test samples
INSTRUMENT="Aeris-fds-Pixcel1d-Medipix3"
WAVELENGTH="Cu"

# --- Step 1: Generate CIFs from DB ---
# Check if CIFs exist to avoid unnecessary re-generation
if [ ! -d "$CIF_DIR" ] || [ -z "$(ls -A $CIF_DIR)" ]; then
    echo "CIF directory is empty. Generating CIFs from $DB_PATH..."
    python scripts/generate_cifs.py
else
    echo "Using existing CIFs in $CIF_DIR"
fi

# --- Step 2: Optimization ---
ray stop --force

# Get total number of physical CPU cores to maximize throughput
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
echo "Detected $NUM_CORES CPU cores. Maximizing DARA parallelization..."

# Set environment variables for Ray and OpenBLAS to prevent over-subscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAY_NUM_CPUS=$NUM_CORES

# --- Step 3: Execution ---
echo "Starting DARA Evaluation on UniqCry TEST set (Seed=7)..."
echo "Database: $DB_PATH"
echo "XRD Data: $NPZ_DIR"

BASE_OUTPUT_DIR="./evaluation_results-300"

for num_phases in 2 3 4
do
    echo "Running evaluation for $num_phases phases..."
    PHASE_OUTPUT_DIR="$BASE_OUTPUT_DIR/${num_phases}_phases"

    python scripts/evaluate_dara_benchmark.py \
        --db_path "$DB_PATH" \
        --npz_dir "$NPZ_DIR" \
        --cif_dir "$CIF_DIR" \
        --output_dir "$PHASE_OUTPUT_DIR" \
        --max_samples "$MAX_SAMPLES" \
        --sample_ratio 0.1 \
        --instrument "$INSTRUMENT" \
        --wavelength "$WAVELENGTH" \
        --num_phases $num_phases
done

echo "Evaluation completed. Results are in $BASE_OUTPUT_DIR"
