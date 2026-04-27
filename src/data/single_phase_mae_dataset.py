"""Dataset for Single-Phase MAE Pre-training."""

import torch
import random
import logging
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional

from .core import scan_files_with_extension, load_npz_pattern, process_pattern

class SinglePhaseMaskedDataset(Dataset):
    """
    Dataset for loading single-phase XRD patterns for MAE pre-training.
    Returns raw or processed intensity tensors.
    """
    
    def __init__(
        self,
        singlephase_xrd_db_path: str,
        xrd_length: int = 3500,
        norm_method: str = "minmax",
        data_fraction: float = 1.0,
        cache_index: bool = True
    ):
        self.db_path = singlephase_xrd_db_path
        self.xrd_length = xrd_length
        self.norm_method = norm_method
        
        # 1. Build Index (Flat list of files)
        logging.info("Scanning single-phase files for pre-training...")
        self.file_paths = scan_files_with_extension(self.db_path, ".npz")
        
        # 2. Downsample if needed
        if data_fraction < 1.0:
            random.seed(42)
            random.shuffle(self.file_paths)
            keep = int(len(self.file_paths) * data_fraction)
            self.file_paths = self.file_paths[:keep]
            
        logging.info(f"MAE Dataset ready: {len(self.file_paths)} samples.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Robust loading with retries
        attempts = 0
        while attempts < 5:
            current_idx = idx if attempts == 0 else random.randint(0, len(self.file_paths)-1)
            path = self.file_paths[current_idx]
            
            pattern = load_npz_pattern(path)
            if pattern is not None:
                # MAE usually expects raw data if normalization is in model,
                # BUT legacy code might expect processing here.
                # Per previous analysis, MAE does normalization in Model.
                # However, padding/truncation MUST happen here.
                
                tensor = process_pattern(
                    pattern, 
                    self.xrd_length, 
                    norm_method="none", # Do not normalize here
                    augment=False
                )
                return tensor
            
            attempts += 1
            
        return torch.zeros(self.xrd_length)
