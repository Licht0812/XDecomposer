"""Distributed Data Parallel (DDP) Utilities."""

import os
import torch
import torch.distributed as dist
from typing import Tuple

def setup_ddp() -> Tuple[int, int, int]:
    """
    Sets up distributed data parallel training environment.
    Returns:
        rank (int): Global rank.
        local_rank (int): Local rank on the node.
        world_size (int): Total number of processes.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method='env://')
            
        dist.barrier()
        return rank, local_rank, world_size
    
    print("⚠️ Not running in DDP mode. Fallback to single GPU.")
    return 0, 0, 1

def cleanup_ddp():
    """Clean up DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
