import os
import random
import logging
import pickle
import numpy as np
import torch
import torch.utils.data as data
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import glob

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnlineMixingConfig:
    """Configuration for Online Mixing XRD Dataset."""
    # Dataset Generation
    MIN_K: int = 2
    MAX_K: int = 4
    MIN_WEIGHT: float = 0.15
    K_DISTRIBUTION: str = "uniform"  # "uniform" or "weighted"
    K_WEIGHTS: Optional[Tuple[float, ...]] = None
    WEIGHT_DISTRIBUTION: str = "random"  # "random" (Dirichlet) or "equal"
    
    # Signal Processing
    XRD_LENGTH: int = 3500
    NORM_METHOD: str = "max"  # "max", "minmax", or "none"
    AUGMENT: bool = True
    NOISE_LEVEL: float = 0.01
    
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
        # 使用 mmap_mode="r" 可以加快网络文件系统的随机读取速度
        with np.load(file_path, mmap_mode="r") as data:
            for key in key_priority:
                if key in data:
                    arr = data[key]
                    if arr.ndim == 1: 
                        arr = arr.astype(np.float32)
                        return np.nan_to_num(arr)
                    elif arr.ndim == 2: 
                        arr = arr[0].astype(np.float32)
                        return np.nan_to_num(arr)
            keys = sorted([k for k in data.keys() if k.startswith('xrd_') or k.startswith('pattern_')])
            if keys: 
                arr = data[keys[0]].astype(np.float32)
                return np.nan_to_num(arr)
            for k in data.keys():
                if k != 'x' and data[k].ndim == 1: 
                    arr = data[k].astype(np.float32)
                    return np.nan_to_num(arr)
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
    
    # Note: For single phase before mixing, norm_method="none" as requested
    if norm_method == "minmax":
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min: tensor = (tensor - t_min) / (t_max - t_min)
    elif norm_method == "max":
        t_max = tensor.max() + 1e-8
        tensor = tensor / t_max
    
    # Noise is added post-mixing in the dataset class
    return tensor

# =============================================================================
# Datasets
# =============================================================================

class XRDBaseDataset(data.Dataset):
    """Base class for managing Single-Phase XRD Database indexing."""
    def __init__(self, singlephase_db_path: str, xrd_length: int = 3500, cache_index: bool = True):
        self.singlephase_db_path = singlephase_db_path
        self.xrd_length = xrd_length
        self.index: Dict[int, List[str]] = {}
        if singlephase_db_path:
            self.index = self._build_or_load_index(cache_index)
            
    def _build_or_load_index(self, use_cache: bool) -> Dict[int, List[str]]:
        """Builds an index of Crystal ID -> List of File Paths."""
        cache_path = os.path.join(self.singlephase_db_path, "crystal_index_v2.pkl")
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f: return pickle.load(f)
            except Exception: pass
        
        logging.info(f"Indexing single-phase database at {self.singlephase_db_path}...")
        index = {}
        
        # Use glob to find all npz files recursively
        all_files = glob.glob(os.path.join(self.singlephase_db_path, "**/*.npz"), recursive=True)
        total_files = len(all_files)
        
        for file_path in all_files:
            name = os.path.basename(file_path)
            try:
                cid = -1
                # 统一的晶体 ID 提取逻辑：支持 crystal_626909_xxx, rruff_626909, 626909 等格式
                if name.startswith("crystal_"):
                    parts = name.split('_')
                    if len(parts) >= 2 and parts[1].isdigit(): 
                        cid = int(parts[1])
                elif name.startswith("rruff_"):
                    parts = name.split('_')
                    if len(parts) >= 2:
                        # 移除可能的后缀（如 .npz）
                        id_str = parts[1].split('.')[0]
                        if id_str.isdigit(): cid = int(id_str)
                else:
                    # 尝试从纯数字文件名中提取
                    id_str = name.split('.')[0]
                    if id_str.isdigit(): cid = int(id_str)
                    elif '_' in id_str:
                        # 处理像 626909_0.npz 这种没有 crystal_ 前缀但有下划线的格式
                        sub_parts = id_str.split('_')
                        if sub_parts[0].isdigit(): cid = int(sub_parts[0])
                
                if cid != -1:
                    if cid not in index:
                        index[cid] = []
                    index[cid].append(file_path)
            except: continue
            
        # 统计索引结果
        unique_crystals = len(index)
        total_indexed_files = sum(len(v) for v in index.values())
        print(f"Indexing complete: Found {unique_crystals} unique crystals and {total_indexed_files}/{total_files} total .npz samples.")
        
        # 过滤掉没有任何有效样本的 ID
        valid_index = {}
        for cid, paths in index.items():
            if paths:
                valid_index[cid] = paths
        
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f: pickle.dump(valid_index, f)
            except Exception: pass
        return valid_index

    def get_crystal_patterns(self, crystal_id: int, max_samples: int = 1) -> List[np.ndarray]:
        """Loads raw patterns for a given crystal ID with data quality validation."""
        paths = self.index.get(crystal_id, [])
        if not paths: return []
        
        # 尝试最多 5 次加载有效样本，防止加载到全 0 或 NaN 的损坏数据
        random.shuffle(paths)
        patterns = []
        for p in paths:
            arr = load_npz_pattern(p)
            if arr is not None and np.nanmax(arr) > 1e-6: # 强度过滤：必须有可见信号
                patterns.append(arr)
                if len(patterns) >= max_samples:
                    break
        return patterns

class OnlineMixingXRDDataset(XRDBaseDataset):
    """
    Dataset that generates multiphase XRD patterns on-the-fly.
    """
    def __init__(
        self,
        singlephase_xrd_db_path: str,
        crystal_ids: Optional[List[int]] = None,
        min_k: int = 2,
        max_k: int = 4,
        min_weight: float = 0.15,
        k_distribution: str = "uniform",
        k_weights: Optional[List[float]] = None,
        weight_distribution: str = "random",
        xrd_length: int = 3500,
        norm_method: str = "max",
        augment: bool = True,
        noise_level: float = 0.01,
        **kwargs
    ):
        super().__init__(singlephase_xrd_db_path, xrd_length, cache_index=True)
        self.min_k, self.max_k = min_k, max_k
        self.min_weight = min_weight
        self.k_weights = self._validate_k_weights(k_weights, k_distribution, min_k, max_k)
        self.weight_distribution = weight_distribution
        self.target_norm_method = norm_method
        self.augment, self.noise_level = augment, noise_level
        
        # Requirement 5: Split by Crystal ID
        if crystal_ids is not None:
            self.indices = np.array(crystal_ids)
        else:
            self.indices = np.array(sorted(list(self.index.keys())))

    @staticmethod
    def _validate_k_weights(weights, dist, min_k, max_k):
        if dist != "weighted": return None
        n = max_k - min_k + 1
        if weights is None or len(weights) != n: return [1.0/n] * n
        s = sum(weights)
        return [w/s for w in weights] if s > 0 else [1.0/n]*n

    def __len__(self) -> int: return len(self.indices)

    def _get_mixing_components(self, anchor_id: int) -> List[Dict[str, Any]]:
        # Requirement 6: Dirichlet weight distribution and min_weight constraint
        k = np.random.choice(range(self.min_k, self.max_k + 1), p=self.k_weights) if self.k_weights else random.randint(self.min_k, self.max_k)
        selected_ids = {anchor_id}
        while len(selected_ids) < k: 
            # Randomly pick from available indices
            other_id = int(np.random.choice(self.indices))
            selected_ids.add(other_id)
            
        pids = list(selected_ids)
        random.shuffle(pids)
        
        if self.weight_distribution == "equal": 
            weights = np.ones(k) / k
        else:
            # Dirichlet with min_weight constraint
            rem = 1.0 - k * self.min_weight
            if rem < 0: 
                weights = np.ones(k) / k
            else:
                raw = np.random.dirichlet(np.ones(k))
                weights = raw * rem + self.min_weight
        return [{'id': pid, 'w': w} for pid, w in zip(pids, weights)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor_id = int(self.indices[idx])
        plan = self._get_mixing_components(anchor_id)
        
        raw_tensors, weights, pids = [], [], []
        for item in plan:
            # Requirement 3: Randomly pick one sample
            patterns = self.get_crystal_patterns(item['id'], max_samples=1)
            # Requirement 4: norm_method="none" before mixing
            t = process_pattern(patterns[0], self.xrd_length, norm_method="none") if patterns else torch.zeros(self.xrd_length)
            raw_tensors.append(t)
            weights.append(item['w'])
            pids.append(item['id'])
            
        # Mix the components
        mix = torch.zeros(self.xrd_length)
        for t, w in zip(raw_tensors, weights): 
            mix += t * w
            
        # Requirement 4: Gaussian noise relative to mix.max()
        if self.augment and self.noise_level > 0:
            noise = torch.randn_like(mix) * self.noise_level * (mix.max() + 1e-8)
            mix = torch.clamp(mix + noise, min=0)
            
        # Requirement 4: Max Scaling post-mixing
        scale = mix.max() + 1e-8 if self.target_norm_method == "max" else 1.0
        mix_norm = mix / scale
        
        # Separated patterns should also be scaled by the same factor
        targets_norm = [(t * w) / scale for t, w in zip(raw_tensors, weights)]
        
        # Pad to max_k for consistent batching
        pad_n = self.max_k - len(targets_norm)
        if pad_n > 0:
            targets_norm.extend([torch.zeros(self.xrd_length)] * pad_n)
            weights.extend([0.0] * pad_n)
            pids.extend([-1] * pad_n)
            
        return {
            'multiphase_xrd': mix_norm.unsqueeze(0), # [1, L]
            'single_xrds': torch.stack(targets_norm), # [K, L]
            'phase_ids': torch.tensor(pids, dtype=torch.long),
            'weights': torch.tensor(weights, dtype=torch.float),
            'num_phases': len(plan)
        }

def get_dataloaders(
    singlephase_xrd_db_path: str,
    config: OnlineMixingConfig,
    batch_size: int = 32,
    num_workers: int = 4
):
    """Creates Train, Val, and Test dataloaders with 8:1:1 split."""
    # Build initial index to get all unique IDs
    base_dataset = XRDBaseDataset(singlephase_xrd_db_path, xrd_length=config.XRD_LENGTH)
    all_ids = sorted(list(base_dataset.index.keys()))
    
    # Requirement 5: Split by ID with fixed seed
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    random.shuffle(all_ids)
    
    n = len(all_ids)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train+n_val]
    test_ids = all_ids[n_train+n_val:]
    
    print(f"Dataset Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    train_dataset = OnlineMixingXRDDataset(
        singlephase_xrd_db_path, 
        crystal_ids=train_ids,
        min_k=config.MIN_K,
        max_k=config.MAX_K,
        min_weight=config.MIN_WEIGHT,
        xrd_length=config.XRD_LENGTH,
        norm_method=config.NORM_METHOD,
        augment=config.AUGMENT,
        noise_level=config.NOISE_LEVEL
    )
    
    val_dataset = OnlineMixingXRDDataset(
        singlephase_xrd_db_path, 
        crystal_ids=val_ids,
        min_k=config.MIN_K,
        max_k=config.MAX_K,
        min_weight=config.MIN_WEIGHT,
        xrd_length=config.XRD_LENGTH,
        norm_method=config.NORM_METHOD,
        augment=False, # No augment for val/test
        noise_level=config.NOISE_LEVEL
    )
    
    test_dataset = OnlineMixingXRDDataset(
        singlephase_xrd_db_path, 
        crystal_ids=test_ids,
        min_k=config.MIN_K,
        max_k=config.MAX_K,
        min_weight=config.MIN_WEIGHT,
        xrd_length=config.XRD_LENGTH,
        norm_method=config.NORM_METHOD,
        augment=False,
        noise_level=config.NOISE_LEVEL
    )
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
