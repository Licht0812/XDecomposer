"""Checkpoint Management Utilities."""

import os
import torch
import logging
from typing import Dict, Any, Optional

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object], # Added scheduler
    epoch: int,
    path: str,
    config: Dict[str, Any],
    rank: int,
    verbose: bool = True
):
    if rank != 0:
        return

    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)
    if verbose:
        logging.info(f"Saved checkpoint to {path}")

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: torch.device = torch.device('cpu'),
    weights_only: bool = False
) -> int:
    """
    Loads model checkpoint.
    Returns:
        start_epoch (int): The next epoch to resume from.
    """
    if not os.path.isfile(path):
        logging.warning(f"Checkpoint {path} not found!")
        return 0

    logging.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    # Load Model
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = 0

    # Load Optimizer, Scheduler & Epoch
    if not weights_only:
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming from epoch {start_epoch}")
    else:
        logging.info("Loaded weights only. Resetting epoch to 0.")

    return start_epoch
