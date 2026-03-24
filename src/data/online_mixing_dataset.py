"""Dataset for Online Mixing XRD."""

import torch
import torch.utils.data as data
import numpy as np
import random
import logging
from typing import List, Dict, Any, Optional

from .core import XRDBaseDataset, process_pattern, XRDCollateFunction
from .config import OnlineMixingConfig

# Alias for backward compatibility
MultiphaseXRDCollateFunction = XRDCollateFunction

class OnlineMixingXRDDataset(XRDBaseDataset):
    """
    Dataset that generates multiphase XRD patterns on-the-fly.
    Inherits scanning capabilities from XRDBaseDataset.
    """

    def __init__(
        self,
        singlephase_xrd_db_path: str,
        crystal_db_path: str = "",
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
        random_single_sample: bool = True,
        **kwargs
    ):
        super().__init__(singlephase_xrd_db_path, xrd_length, cache_index=True)
        
        self.min_k = min_k
        self.max_k = max_k
        self.min_weight = min_weight
        self.k_weights = self._validate_k_weights(k_weights, k_distribution, min_k, max_k)
        self.weight_distribution = weight_distribution
        self.target_norm_method = norm_method
        self.augment = augment
        self.noise_level = noise_level
        self.random_single_sample = random_single_sample

        # Indices
        if crystal_ids is not None:
            self.indices = np.array(crystal_ids)
        else:
            self.indices = np.array(sorted(list(self.index.keys())))
            
        logging.info(f"Online Mixing Dataset: {len(self.indices)} anchors, K=[{min_k}-{max_k}]")

    @staticmethod
    def _validate_k_weights(weights, dist, min_k, max_k):
        if dist != "weighted": return None
        n = max_k - min_k + 1
        if weights is None or len(weights) != n:
            return [1.0/n] * n
        s = sum(weights)
        return [w/s for w in weights] if s > 0 else [1.0/n]*n

    def __len__(self) -> int:
        return len(self.indices)

    def _get_mixing_components(self, anchor_id: int) -> List[Dict[str, Any]]:
        # 1. Choose K
        if self.k_weights:
            k = np.random.choice(range(self.min_k, self.max_k + 1), p=self.k_weights)
        else:
            k = random.randint(self.min_k, self.max_k)
            
        # 2. Choose IDs
        selected_ids = {anchor_id}
        while len(selected_ids) < k:
            selected_ids.add(int(np.random.choice(self.indices)))
        pids = list(selected_ids)
        random.shuffle(pids) # Shuffle to randomize position
        
        # 3. Choose Weights
        if self.weight_distribution == "equal":
            weights = np.ones(k) / k
        else:
            rem = 1.0 - k * self.min_weight
            if rem < 0: weights = np.ones(k) / k
            else:
                raw = np.random.dirichlet(np.ones(k))
                weights = raw * rem + self.min_weight
                
        return [{'id': pid, 'w': w, 'anchor': pid==anchor_id} for pid, w in zip(pids, weights)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor_id = int(self.indices[idx])
        plan = self._get_mixing_components(anchor_id)
        
        raw_tensors = []
        weights = []
        pids = []
        
        # Load & Process
        for item in plan:
            # Load Raw (No Norm)
            patterns = self.get_crystal_patterns(item['id'], max_samples=1)
            if patterns:
                # Basic processing (pad/crop) only
                t = process_pattern(patterns[0], self.xrd_length, norm_method="none")
            else:
                t = torch.zeros(self.xrd_length)
                
            raw_tensors.append(t)
            weights.append(item['w'])
            pids.append(item['id'])
            
        # Mix
        mix = torch.zeros(self.xrd_length)
        for t, w in zip(raw_tensors, weights):
            mix += t * w
            
        # Noise
        if self.augment and self.noise_level > 0:
            noise = torch.randn_like(mix) * self.noise_level * (mix.max() + 1e-8)
            mix = torch.clamp(mix + noise, min=0)
            
        # Normalize (Max Scaling)
        scale = 1.0
        if self.target_norm_method == "max":
            scale = mix.max() + 1e-8
            
        mix_norm = mix / scale
        targets_norm = [(t * w) / scale for t, w in zip(raw_tensors, weights)]
        
        # Pad to max_k
        pad_n = self.max_k - len(targets_norm)
        if pad_n > 0:
            zeros = torch.zeros(self.xrd_length)
            targets_norm.extend([zeros] * pad_n)
            weights.extend([0.0] * pad_n)
            pids.extend([-1] * pad_n)
            
        targets_stack = torch.stack(targets_norm)
        
        return {
            'multiphase_xrd': mix_norm,
            'single_xrds': targets_stack,
            'concatenated_xrds': torch.cat([mix_norm.unsqueeze(0), targets_stack], dim=0),
            'phase_ids': torch.tensor(pids, dtype=torch.long),
            'weights': torch.tensor(weights, dtype=torch.float),
            'sample_idx': idx,
            'anchor_id': anchor_id,
            'num_phases': len(plan)
        }

# Helper for DataLoader creation
def create_online_mixing_dataloader(
    singlephase_xrd_db_path: str,
    crystal_db_path: str,
    config: Optional[OnlineMixingConfig] = None, # Add config object support
    split: str = 'train',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    distributed: bool = False,
    **kwargs
) -> data.DataLoader:
    
    # Use default config if not provided
    if config is None:
        config = OnlineMixingConfig()
    
    # Use config values if not overridden in kwargs
    # Logic: kwargs > config > defaults
    
    # 1. Dataset Params
    dataset_kwargs = {
        'min_k': kwargs.get('min_k', config.MIN_K),
        'max_k': kwargs.get('max_k', config.MAX_K),
        'min_weight': kwargs.get('min_weight', config.MIN_WEIGHT),
        'k_distribution': kwargs.get('k_distribution', config.K_DISTRIBUTION),
        'k_weights': kwargs.get('k_weights', config.K_WEIGHTS),
        'weight_distribution': kwargs.get('weight_distribution', config.WEIGHT_DISTRIBUTION),
        'xrd_length': kwargs.get('xrd_length', config.XRD_LENGTH),
        'norm_method': kwargs.get('norm_method', config.NORM_METHOD),
        'augment': kwargs.get('augment', config.AUGMENT),
        'noise_level': kwargs.get('noise_level', config.NOISE_LEVEL),
        'random_single_sample': kwargs.get('random_single_sample', config.RANDOM_SINGLE_SAMPLE)
    }
    
    # Use base class to quick-scan IDs
    temp_ds = XRDBaseDataset(singlephase_xrd_db_path)
    all_ids = sorted(list(temp_ds.index.keys()))
    
    # Split using config seed
    seed = kwargs.get('seed', config.SEED)
    np.random.seed(seed)
    shuffled = np.array(all_ids)
    np.random.shuffle(shuffled)
    
    # Split ratios from config if not provided
    t_ratio = kwargs.get('train_ratio', train_ratio)
    if 'train_ratio' not in kwargs: t_ratio = config.TRAIN_RATIO
    
    v_ratio = kwargs.get('val_ratio', val_ratio)
    if 'val_ratio' not in kwargs: v_ratio = config.VAL_RATIO
    
    n_train = int(len(shuffled) * t_ratio)
    n_val = int(len(shuffled) * (t_ratio + v_ratio))
    
    if split == 'train': ids = shuffled[:n_train]
    elif split == 'val': ids = shuffled[n_train:n_val]
    else: ids = shuffled[n_val:]
    
    ds = OnlineMixingXRDDataset(
        singlephase_xrd_db_path=singlephase_xrd_db_path,
        crystal_ids=ids.tolist(),
        **dataset_kwargs
    )
    
    sampler = None
    if distributed:
        sampler = data.distributed.DistributedSampler(ds, shuffle=(split=='train'))
        shuffle = False # Sampler handles shuffling
    else:
        shuffle = (split=='train')

    return data.DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=XRDCollateFunction(),
        pin_memory=torch.cuda.is_available()
    )
