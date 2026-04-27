"""Core utilities and base classes for XRD data processing."""

import os
import random
import logging
import pickle
import numpy as np
import torch
import torch.utils.data as data
from typing import List, Dict, Any, Optional, Union

# =============================================================================
# Utils
# =============================================================================

def scan_files_with_extension(directory: str, extension: str = ".npz") -> List[str]:
    """Scans a directory for files with a specific extension."""
    if not os.path.exists(directory):
        logging.warning(f"Directory not found: {directory}")
        return []
    
    files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith(extension):
                files.append(entry.path)
    return sorted(files)

def load_npz_pattern(file_path: str, key_priority: List[str] = None) -> Optional[np.ndarray]:
    """Loads an XRD pattern from an .npz file safely."""
    if not os.path.exists(file_path):
        return None
        
    if key_priority is None:
        key_priority = ['y', 'intensity', 'xrd_pattern', 'pattern', 'xrd']
        
    try:
        with np.load(file_path) as data:
            # 1. Try explicit keys
            for key in key_priority:
                if key in data:
                    arr = data[key]
                    if arr.ndim == 1:
                        return arr.astype(np.float32)
                    elif arr.ndim == 2: # Return first if batch
                        return arr[0].astype(np.float32)

            # 2. Try 'xrd_0', 'pattern_0' etc.
            keys = sorted([k for k in data.keys() if k.startswith('xrd_') or k.startswith('pattern_')])
            if keys:
                return data[keys[0]].astype(np.float32)
                
            # 3. Fallback: First array that looks like data (not 'x')
            for k in data.keys():
                if k != 'x' and data[k].ndim == 1:
                    return data[k].astype(np.float32)
                    
            return None
    except Exception as e:
        # logging.debug(f"Failed to load {file_path}: {e}")
        return None

def process_pattern(
    pattern: np.ndarray, 
    length: int, 
    norm_method: str = "none",
    augment: bool = False,
    noise_level: float = 0.0
) -> torch.Tensor:
    """Processes a raw numpy pattern into a standardized tensor."""
    # 1. To Tensor
    tensor = torch.from_numpy(pattern).float()
    
    # 2. Pad/Crop
    curr_len = tensor.shape[0]
    if curr_len > length:
        tensor = tensor[:length]
    elif curr_len < length:
        padding = torch.zeros(length - curr_len)
        tensor = torch.cat([tensor, padding])
        
    # 3. Augment (Noise)
    if augment and noise_level > 0:
        noise = torch.randn_like(tensor) * noise_level * (tensor.max() + 1e-8)
        tensor = tensor + noise
        tensor = torch.clamp(tensor, min=0)
        
    # 4. Normalize
    if norm_method == "minmax":
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
    elif norm_method == "max":
        t_max = tensor.max() + 1e-8
        tensor = tensor / t_max
        
    return tensor

# =============================================================================
# Collate Function
# =============================================================================

class XRDCollateFunction:
    """Standard Collate function to stack batches."""
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
            
        keys = batch[0].keys()
        result = {}
        
        for k in keys:
            items = [b[k] for b in batch]
            if isinstance(items[0], torch.Tensor):
                result[k] = torch.stack(items, dim=0)
            else:
                result[k] = items
                
        return result

# =============================================================================
# Base Class
# =============================================================================

class XRDBaseDataset(data.Dataset):
    """Base class for managing Single-Phase XRD Database indexing."""
    
    def __init__(
        self, 
        singlephase_db_path: str, 
        xrd_length: int = 3500,
        cache_index: bool = True
    ):
        self.singlephase_db_path = singlephase_db_path
        self.xrd_length = xrd_length
        self.index: Dict[int, List[str]] = {}
        
        if singlephase_db_path:
            self.index = self._build_or_load_index(cache_index)
            
    def _build_or_load_index(self, use_cache: bool) -> Dict[int, List[str]]:
        """Builds an index of Crystal ID -> List of File Paths."""
        cache_path = os.path.join(self.singlephase_db_path, "crystal_index.pkl")
        
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
                
        logging.info("Indexing single-phase database...")
        index = {}
        with os.scandir(self.singlephase_db_path) as entries:
            for entry in entries:
                if not entry.name.endswith('.npz'):
                    continue
                    
                # Parsing logic: crystal_{id}_... or {id}.npz
                try:
                    name = entry.name
                    cid = -1
                    if name.startswith("crystal_"):
                        parts = name.split('_')
                        if len(parts) >= 2 and parts[1].isdigit():
                            cid = int(parts[1])
                    elif name[:-4].isdigit():
                        cid = int(name[:-4])
                        
                    if cid != -1:
                        if cid not in index: index[cid] = []
                        index[cid].append(entry.path)
                except:
                    continue
                    
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(index, f)
            except Exception:
                pass
                
        return index

    def get_crystal_patterns(self, crystal_id: int, max_samples: int = 1) -> List[np.ndarray]:
        """Loads raw patterns for a given crystal ID."""
        paths = self.index.get(crystal_id, [])
        if not paths:
            return []
            
        selected_paths = random.sample(paths, min(len(paths), max_samples))
        patterns = []
        for p in selected_paths:
            arr = load_npz_pattern(p)
            if arr is not None:
                patterns.append(arr)
        return patterns

    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise NotImplementedError
