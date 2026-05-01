"""Distributed Data Parallel (DDP) Utilities."""

import os
import socket
import torch
import torch.distributed as dist
from typing import Tuple

def setup_ddp() -> Tuple[int, int, int]:
    """
    Sets up distributed data parallel training environment.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        try:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
        except RuntimeError as exc:
            raise RuntimeError(
                "CUDA initialization failed before DDP setup "
                f"on host={socket.gethostname()}, rank={rank}, local_rank={local_rank}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            ) from exc

        if not cuda_available or local_rank >= device_count:
            raise RuntimeError(
                "DDP requested CUDA/NCCL but the expected local GPU is not visible "
                f"on host={socket.gethostname()}, rank={rank}, local_rank={local_rank}, "
                f"world_size={world_size}, cuda_available={cuda_available}, "
                f"device_count={device_count}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )

        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method='env://',
                    device_id=torch.device(f"cuda:{local_rank}")
                )
            except TypeError:
                dist.init_process_group(backend="nccl", init_method='env://')

        return rank, local_rank, world_size

    print("Not running in DDP mode. Fallback to single GPU.")
    return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
