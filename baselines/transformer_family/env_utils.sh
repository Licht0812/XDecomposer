#!/bin/bash
# Shared environment and distributed-launch helpers for transformer_family scripts.

activate_spectra_env() {
    require_conda_env
}

parse_slurm_gpu_count() {
    local value="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
    if [ -n "$value" ]; then
        if [[ "$value" == *"("* ]]; then
            value="${value%%(*}"
        fi
        if [[ "$value" == *":"* ]]; then
            value="${value##*:}"
        fi
        if [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "$value"
            return
        fi
    fi

    value="${SLURM_TRES_PER_NODE:-${SLURM_JOB_GRES:-}}"
    if [[ "$value" =~ gres/gpu[:=]([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    if [[ "$value" =~ gpu[:=]([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi

    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        local total_gpus
        total_gpus=$(count_csv_items "$SLURM_JOB_GPUS")
        if [ "${NNODES:-1}" -gt 1 ] && [ "$total_gpus" -gt "${NNODES:-1}" ] && [ $((total_gpus % NNODES)) -eq 0 ]; then
            echo $((total_gpus / NNODES))
        else
            echo "$total_gpus"
        fi
        return
    fi

    echo 1
}

configure_gpu_layout() {
    NNODES=1
    if [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
        NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
        NUM_GPUS=$(parse_slurm_gpu_count)
        if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [[ "$SLURM_GPUS_ON_NODE" =~ ^[0-9]+$ ]] && [ "$NUM_GPUS" -gt "$SLURM_GPUS_ON_NODE" ]; then
            echo "Parsed NUM_GPUS=$NUM_GPUS exceeds SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE" >&2
            exit 1
        fi
    else
        if [ -n "${GPUS:-}" ]; then
            NUM_GPUS=$(count_csv_items "$GPUS")
            export CUDA_VISIBLE_DEVICES="$GPUS"
        else
            NUM_GPUS=$(detect_num_gpus)
        fi
    fi
}

configure_distributed_env() {
    MASTER_ADDR="localhost"
    if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    fi
    MASTER_PORT="$PORT"
    export NUM_GPUS NNODES MASTER_ADDR MASTER_PORT
}

check_node_cuda_visibility() {
    echo "CUDA preflight on $(hostname): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    nvidia-smi -L
    python -c 'import os, socket, torch; print("torch cuda preflight:", "host=" + socket.gethostname(), "CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), "is_available=" + str(torch.cuda.is_available()), "device_count=" + str(torch.cuda.device_count()))'
}

run_cuda_preflight() {
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "$NNODES" -gt 1 ]; then
        export -f check_node_cuda_visibility
        srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
            check_node_cuda_visibility
        '
    else
        check_node_cuda_visibility
    fi
}

launch_torchrun() {
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "$NNODES" -gt 1 ]; then
        srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
            python -m torch.distributed.run \
                --nnodes="$NNODES" \
                --nproc_per_node="$NUM_GPUS" \
                --node_rank="$SLURM_PROCID" \
                --master_addr="$MASTER_ADDR" \
                --master_port="$MASTER_PORT" \
                "$@"
        ' bash "$@"
    else
        python -m torch.distributed.run \
            --nnodes=1 \
            --nproc_per_node="$NUM_GPUS" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            "$@"
    fi
}
