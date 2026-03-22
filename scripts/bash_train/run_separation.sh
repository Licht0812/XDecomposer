#!/bin/bash
#SBATCH -J mix_crystal_sep
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -p project1
#SBATCH -A project1
#SBATCH -o logs/sep_%j.out
#SBATCH -e logs/sep_%j.err
set -e

# ==========================================
# Separation Training
# ==========================================

source "$(dirname "$0")/../../configs/paths.sh"

GPUS="0,1"
PORT=$((29500 + $RANDOM % 100))
EXP_NAME="separation"
RESUME_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        *)
            if [ -z "$RESUME_PATH" ] && [[ "$1" != -* ]]; then
                RESUME_PATH="$1"
                shift 1
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
cd "$(dirname "$0")/../.."

if [ -n "$SLURM_JOB_NUM_NODES" ]; then
    NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
else
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    export CUDA_VISIBLE_DEVICES=$GPUS
fi

echo "============================================"
echo "Experiment: $EXP_NAME"
echo "GPUs: $GPUS (Count: $NUM_GPUS)"
echo "Port: $PORT"
if [ -n "$RESUME_PATH" ]; then
    echo "Resume: $RESUME_PATH"
fi
echo "============================================"

OUTPUT_DIR="checkpoints/hybrid_${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

RESUME_ARG=""
if [ -n "$RESUME_PATH" ]; then
    CLEAN_PATH="$RESUME_PATH"
    if [[ "$RESUME_PATH" == *".ptbash" ]]; then
        CLEAN_PATH="${RESUME_PATH%bash}"
    fi
    RESUME_ARG="--resume $CLEAN_PATH"
    OUTPUT_DIR=$(dirname "$CLEAN_PATH")
fi

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    --rdzv_endpoint=localhost:$PORT \
    scripts/python_runners/train_separation_film.py \
    --experiment_name "$EXP_NAME" \
    --save_dir "$OUTPUT_DIR" \
    $RESUME_ARG \
    --singlephase_xrd_db "$PATH_DATA_SINGLEPHASE" \
    --xrd_length 3500 \
    --num_phases 4 \
    --batch_size 128 \
    --augment \
    --mae_checkpoint "$PATH_CKPT_MAE" \
    --cnn_channels 48 96 192 384 \
    --cnn_kernels 15 8 8 10 \
    --cnn_strides 1 2 2 5 \
    --epochs 2000 \
    --lr 2e-4 \
    --lr_scheduler "cosine" \
    --warmup_epochs 20 \
    --noam_factor 1.0 \
    --alpha 5.0 \
    --lambda_sisdr 0.5 \
    --lambda_geo 5.0 \
    --beta 2 \
    --lambda_tv 0 \
    --lambda_mix 5 \
    --lambda_activity 2.0 \
    --activity_threshold 0.8 \
    --vis_interval 50
