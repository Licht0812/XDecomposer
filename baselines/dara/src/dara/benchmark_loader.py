"""
Benchmark Data Loader for DARA.
Handles .npz loading, online mixing, and dataset splitting.
Adapted from user reference.
"""

import os
import random
import logging
import pickle
import numpy as np
import torch
import torch.utils.data as data
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---

@dataclass
class OnlineMixingConfig:
    """Configuration for Online Mixing XRD Dataset."""
    # Dataset Generation
    MIN_K: int = 2
    MAX_K: int = 4
    MIN_WEIGHT: float = 0.15
    K_DISTRIBUTION: str = "weighted"  # "uniform" or "weighted"
    K_WEIGHTS: Tuple[float, ...] = (0.5, 0.3, 0.2) # Updated for 2, 3, 4 phases
    WEIGHT_DISTRIBUTION: str = "random"  # "random" (Dirichlet) or "equal"
    
    # Signal Processing
    XRD_LENGTH: int = 3500
    NORM_METHOD: str = "max"  # "max", "minmax", or "none"
    AUGMENT: bool = False
    NOISE_LEVEL: float = 0.001
    
    # Paths for UniqCry Dataset
    DB_PATH: str = "/data/group/project1/Crystal/UniqCryLabeled.db"
    NPZ_DIR: str = "/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data/"
    CIF_DIR: str = "/data/home/zdhs0019/Projects/xrd_baselines/dara/dataset/uniqcry_cifs"
    
    # Optimization
    RANDOM_SINGLE_SAMPLE: bool = True
    
    # Splitting
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    SEED: int = 7

# =============================================================================
# Core Utilities
# =============================================================================

def load_npz_pattern(file_path: str, key_priority: List[str] = None) -> Optional[np.ndarray]:
    """Loads an XRD pattern from an .npz file safely."""
    if not os.path.exists(file_path):
        return None
    if key_priority is None:
        key_priority = ['y', 'intensity', 'xrd_pattern', 'pattern', 'xrd']
    try:
        with np.load(file_path) as data:
            for key in key_priority:
                if key in data:
                    arr = data[key]
                    if arr.ndim == 1: return arr.astype(np.float32)
                    elif arr.ndim == 2: return arr[0].astype(np.float32)
            # Fallback to any 1D array
            for k in data.keys():
                if k != 'x' and data[k].ndim == 1: return data[k].astype(np.float32)
            return None
    except Exception:
        return None

def process_pattern(
    pattern: np.ndarray, 
    length: int, 
    norm_method: str = "none",
    augment: bool = False,
    noise_level: float = 0.0
) -> torch.Tensor:
    """Processes a raw numpy pattern into a standardized tensor."""
    tensor = torch.from_numpy(pattern).float()
    curr_len = tensor.shape[0]
    if curr_len > length: tensor = tensor[:length]
    elif curr_len < length:
        padding = torch.zeros(length - curr_len)
        tensor = torch.cat([tensor, padding])
    if augment and noise_level > 0:
        noise = torch.randn_like(tensor) * noise_level * (tensor.max() + 1e-8)
        tensor = torch.clamp(tensor + noise, min=0)
    if norm_method == "minmax":
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min: tensor = (tensor - t_min) / (t_max - t_min)
    elif norm_method == "max":
        t_max = tensor.max() + 1e-8
        tensor = tensor / t_max
    return tensor

class XRDBaseDataset(data.Dataset):
    """Base class for managing Single-Phase XRD Database indexing."""
    def __init__(self, singlephase_db_path: str, xrd_length: int = 3500, cache_index: bool = True):
        self.singlephase_db_path = singlephase_db_path
        self.xrd_length = xrd_length
        self.index: Dict[int, List[str]] = {}
        if singlephase_db_path:
            self.index = self._build_or_load_index(cache_index)
            
    def _build_or_load_index(self, use_cache: bool) -> Dict[int, List[str]]:
        """Builds an index of Crystal ID -> One Valid File Path."""
        cache_path = os.path.join(self.singlephase_db_path, "crystal_index_v2.pkl")
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        logging.info(f"Indexing single-phase database at {self.singlephase_db_path}...")
        index = {}
        
        if not os.path.exists(self.singlephase_db_path):
            logging.error(f"Directory NOT FOUND: {self.singlephase_db_path}")
            return {}

        # Group files by crystal ID first
        files_by_cid = {}
        npz_count = 0
        import re
        
        # Pattern to match crystal_{Label}_sample_{num}.npz
        pattern = re.compile(r'crystal_(\d+)_sample_(\d+)\.npz')
        
        for root, _, files in os.walk(self.singlephase_db_path):
            for name in files:
                match = pattern.match(name)
                if not match:
                    continue
                
                npz_count += 1
                try:
                    # Label is extracted from the filename
                    label = int(match.group(1))
                    # The standard ID in UniqCryLabeled.db is Label + 1
                    cid = label + 1
                    if cid not in files_by_cid:
                        files_by_cid[cid] = []
                    files_by_cid[cid].append(os.path.join(root, name))
                except:
                    continue

        logging.info(f"Found {npz_count} .npz files for {len(files_by_cid)} unique crystal IDs.")

        # For each crystal ID, find exactly one valid sample
        valid_index = {}
        if not files_by_cid:
            logging.error("No valid crystal IDs extracted from filenames.")
            return {}

        # For each crystal ID, find exactly one valid sample
        valid_index = {}
        for cid, paths in tqdm(files_by_cid.items(), desc="Validating crystal patterns"):
            # Randomly shuffle paths to pick a random one, then check validity
            random.shuffle(paths)
            found_valid = False
            for p in paths:
                pattern = load_npz_pattern(p)
                if pattern is not None and pattern.size > 0 and np.max(pattern) > 0:
                    valid_index[cid] = [p] # Keep it as a list for compatibility
                    found_valid = True
                    break
            
            if not found_valid:
                logging.debug(f"No valid patterns found for crystal {cid}")
        
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(valid_index, f)
            except Exception:
                pass
        return valid_index

    def get_crystal_patterns(self, crystal_id: int, max_samples: int = 1) -> List[np.ndarray]:
        """Loads raw patterns for a given crystal ID."""
        paths = self.index.get(crystal_id, [])
        if not paths: return []
        # Max 20 samples logic: paths already contains all samples for this crystal
        selected_paths = random.sample(paths, min(len(paths), max_samples))
        patterns = []
        for p in selected_paths:
            arr = load_npz_pattern(p)
            if arr is not None: patterns.append(arr)
        return patterns

class OnlineMixingXRDDataset(XRDBaseDataset):
    """
    Dataset that generates multiphase XRD patterns on-the-fly for UniqCry.
    """
    def __init__(
        self, 
        singlephase_xrd_db_path: str,
        crystal_ids: Optional[List[int]] = None,
        min_k: int = 2,
        max_k: int = 4,
        min_weight: float = 0.1,
        k_distribution: str = "weighted",
        k_weights: Optional[List[float]] = None,
        weight_distribution: str = "random",
        xrd_length: int = 3500,
        norm_method: str = "max",
        augment: bool = False,
        noise_level: float = 0.01,
        two_theta_range: Tuple[float, float] = (5.0, 90.0),
        **kwargs
    ):
        super().__init__(singlephase_xrd_db_path, xrd_length, cache_index=True)
        self.min_k, self.max_k = min_k, max_k
        self.min_weight = min_weight
        self.k_weights = self._validate_k_weights(k_weights, k_distribution, min_k, max_k)
        self.weight_distribution = weight_distribution
        self.target_norm_method = norm_method
        self.augment, self.noise_level = augment, noise_level
        
        # Filter indices to only include crystal IDs that have patterns
        available_ids = set(self.index.keys())
        if crystal_ids is not None:
            self.indices = np.array([cid for cid in crystal_ids if cid in available_ids])
        else:
            self.indices = np.array(sorted(list(available_ids)))
            
        self.two_theta_range = two_theta_range

    @staticmethod
    def _validate_k_weights(weights, dist, min_k, max_k):
        if dist != "weighted": return None
        n = max_k - min_k + 1
        if weights is None or len(weights) != n: return [1.0/n] * n
        s = sum(weights)
        return [w/s for w in weights] if s > 0 else [1.0/n]*n

    def __len__(self) -> int: return len(self.indices)

    def _get_mixing_components(self, anchor_id: int) -> List[Dict[str, Any]]:
        k = np.random.choice(range(self.min_k, self.max_k + 1), p=self.k_weights) if self.k_weights else random.randint(self.min_k, self.max_k)
        selected_ids = {anchor_id}
        while len(selected_ids) < k: 
            candidate = int(np.random.choice(self.indices))
            selected_ids.add(candidate)
        pids = list(selected_ids)
        random.shuffle(pids)
        if self.weight_distribution == "equal": weights = np.ones(k) / k
        else:
            rem = 1.0 - k * self.min_weight
            if rem < 0: weights = np.ones(k) / k
            else:
                raw = np.random.dirichlet(np.ones(k))
                weights = raw * rem + self.min_weight
        return [{'id': pid, 'w': w} for pid, w in zip(pids, weights)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor_id = int(self.indices[idx])
        plan = self._get_mixing_components(anchor_id)
        raw_tensors, weights, pids = [], [], []
        for item in plan:
            # Randomly pick one of the 20 samples for this crystal
            patterns = self.get_crystal_patterns(item['id'], max_samples=1)
            t = process_pattern(patterns[0], self.xrd_length, norm_method="none") if patterns else torch.zeros(self.xrd_length)
            raw_tensors.append(t)
            weights.append(item['w'])
            pids.append(item['id'])
        
        mix = torch.zeros(self.xrd_length)
        for t, w in zip(raw_tensors, weights): mix += t * w
        
        if self.augment and self.noise_level > 0:
            noise = torch.randn_like(mix) * self.noise_level * (mix.max() + 1e-8)
            mix = torch.clamp(mix + noise, min=0)
            
        scale = mix.max() + 1e-8 if self.target_norm_method == "max" else 1.0
        mix_norm = mix / scale
        targets_norm = [(t * w) / scale for t, w in zip(raw_tensors, weights)]
        
        pad_n = self.max_k - len(targets_norm)
        if pad_n > 0:
            targets_norm.extend([torch.zeros(self.xrd_length)] * pad_n)
            weights.extend([0.0] * pad_n)
            pids.extend([-1] * pad_n)
            
        return {
            'multiphase_xrd': mix_norm,
            'single_xrds': torch.stack(targets_norm),
            'phase_ids': torch.tensor(pids, dtype=torch.long),
            'weights': torch.tensor(weights, dtype=torch.float),
            'num_phases': len(plan),
            'two_theta': np.linspace(self.two_theta_range[0], self.two_theta_range[1], self.xrd_length)
        }

    def save_to_xy(self, idx: int, out_path: str):
        """Saves a mixed pattern to .xy format for DARA consumption."""
        item = self.__getitem__(idx)
        x = item['two_theta']
        # Scale intensity to a larger range (e.g. 0-10000) 
        # because BGMN/eflech can fail with too small values [0, 1]
        y = item['multiphase_xrd'].numpy() * 10000.0
        np.savetxt(out_path, np.column_stack((x, y)), fmt="%.6f")
        return item

def get_splits_from_db(
    db_path: str,
    config: OnlineMixingConfig = OnlineMixingConfig()
) -> Tuple[List[int], List[int], List[int]]:
    """Generates 8:1:1 splits based on Crystal ID from the SQLite DB."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Query specifically from 'systems' table if it exists
    try:
        cursor.execute("SELECT DISTINCT id FROM systems")
        all_ids = [row[0] for row in cursor.fetchall()]
    except Exception:
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            table_name = tables[0][0]
            cursor.execute(f"SELECT DISTINCT id FROM {table_name}")
            all_ids = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Error reading IDs from DB: {e}. Fallback to numeric scan.")
            all_ids = list(range(1, 15000))
    finally:
        conn.close()
        
    np.random.seed(config.SEED)
    shuffled = np.array(all_ids)
    np.random.shuffle(shuffled)
    n_train = int(len(shuffled) * config.TRAIN_RATIO)
    n_val = int(len(shuffled) * (config.TRAIN_RATIO + config.VAL_RATIO))
    return shuffled[:n_train].tolist(), shuffled[n_train:n_val].tolist(), shuffled[n_val:].tolist()
