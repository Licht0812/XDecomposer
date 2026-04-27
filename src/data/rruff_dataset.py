import sqlite3
import numpy as np
import torch
import torch.utils.data as data
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import ast
import os
import glob
from tqdm import tqdm

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    y = np.asarray(y)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D.dot(D.transpose()) * lam
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

class RRUFFOnlineMixingDataset(data.Dataset):
    def __init__(self, rruff_db_path, min_k=2, max_k=4, k_weights=None, min_weight=0.1, weight_distribution="random", target_length=3500, theta_min=10.0, theta_max=80.0, split='all', num_folds=5, fold=0, seed=42, virtual_epoch_length=1000):
        self.rruff_db_path = rruff_db_path
        self.min_k = min_k
        self.max_k = max_k
        self.k_weights = k_weights
        self.min_weight = min_weight
        self.weight_distribution = weight_distribution
        self.target_length = target_length
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.target_angles = np.linspace(theta_min, theta_max, target_length)
        self.virtual_epoch_length = max(1, int(virtual_epoch_length))
        self.set_epoch_seed(seed)
        
        # Load all valid phase data into memory
        self.phases = [] # list of (id, rruff_id, int_array)
        
        # Fast load from preprocessed directory if it exists
        preprocessed_dir = os.path.join(os.path.dirname(rruff_db_path), "rruff_processed")
        if os.path.isdir(preprocessed_dir) and len(glob.glob(os.path.join(preprocessed_dir, "*.npz"))) > 100:
            print(f"Loading preprocessed RRUFF npz files from {preprocessed_dir}...")
            self._load_preprocessed(preprocessed_dir)
        else:
            print("Processing from standard Database (This will take a while for ALS baseline removal)...")
            self._load_db()
            
        print(f"Loaded {len(self.phases)} total valid XRD patterns from RRUFF before splitting.")
        
        # Determine K-Fold Split
        if split != 'all':
            # Create a deterministically shuffled list of original mineral phases to prevent data leakage in K-Fold
            # Sort phases by their s_id first to ensure determinism across different runtimes
            self.phases.sort(key=lambda x: x[0]) 
            
            rng = random.Random(seed)
            indices = list(range(len(self.phases)))
            rng.shuffle(indices)
            
            fold_size = len(indices) // num_folds
            val_indices = set(indices[fold * fold_size : (fold + 1) * fold_size if fold < num_folds - 1 else len(indices)])
            
            if split == 'train':
                self.phases = [self.phases[i] for i in range(len(self.phases)) if i not in val_indices]
            elif split == 'test' or split == 'val':
                self.phases = [self.phases[i] for i in range(len(self.phases)) if i in val_indices]
                
            print(f"[{split.upper()} Fold {fold}/{num_folds}] Maintained {len(self.phases)} valid base patterns for random mixing.")
        
    def _load_preprocessed(self, prep_dir):
        npz_files = glob.glob(os.path.join(prep_dir, "rruff_*.npz"))
        for f in tqdm(npz_files, desc="Loading Cache"):
            fname = os.path.basename(f)
            s_id = int(fname.split("_")[1].split(".")[0])
            rruff_id = f"ID_{s_id}"
            try:
                data = np.load(f)
                intensities = data['y']
                self.phases.append((s_id, rruff_id, intensities.astype(np.float32)))
            except:
                pass

    def _load_db(self):
        from ase.db import connect
        db = connect(self.rruff_db_path)
        for row in tqdm(db.select(), desc="Processing ASE DB", total=db.count()):
            s_id = row.id
            data_dict = row.data
            if not data_dict or "angle" not in data_dict or "intensity" not in data_dict:
                continue
            
            try:
                angles = np.array(data_dict["angle"])
                intensities = np.array(data_dict["intensity"])
                
                try:
                    baseline = baseline_als(intensities)
                    intensities = intensities - baseline
                    intensities = np.clip(intensities, 0, None)
                except:
                    pass

                f = interp1d(angles, intensities, bounds_error=False, fill_value=0.0)
                std_intensities = f(self.target_angles)
                std_intensities = np.clip(std_intensities, 0, None)

                if std_intensities.max() > 0:
                    std_intensities = std_intensities / std_intensities.max()
                    
                rruff_id = row.get("rruff_id", f"ID_{s_id}")
                if isinstance(rruff_id, str):
                    rruff_id = rruff_id.strip('"')
                self.phases.append((s_id, rruff_id, std_intensities.astype(np.float32)))
            except Exception as e:
                print(f"Error processing {s_id}: {e}")
                continue


    def set_epoch_seed(self, seed):
        self.seed = int(seed)
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    def __len__(self):
        return self.virtual_epoch_length

    def __getitem__(self, idx):
        # Determine number of phases k for this sample
        if self.k_weights is None:
            k = self.py_rng.randint(self.min_k, self.max_k)
        else:
            k = self.py_rng.choices(range(self.min_k, self.max_k + 1), weights=self.k_weights)[0]
            
        chosen = self.py_rng.sample(self.phases, k)
        
        # Weights (aligned to MP20)
        if self.weight_distribution == "equal":
            weights = np.ones(k) / k
        else:
            rem = 1.0 - k * self.min_weight
            if rem < 0: 
                weights = np.ones(k) / k
            else:
                raw = self.np_rng.dirichlet(np.ones(k))
                weights = raw * rem + self.min_weight
        
        mixed_intensity = np.zeros(self.target_length, dtype=np.float32)
        base_intensities = []
        base_ids = []
        base_weights = []
        
        for i, (p_id, r_id, p_intensity) in enumerate(chosen):
            mixed_intensity += weights[i] * p_intensity
            base_intensities.append(p_intensity * weights[i])
            base_ids.append(p_id)
            base_weights.append(float(weights[i]))
            
        # Normalize (Scale both mixture and components to match MP20)
        scale = 1.0
        if mixed_intensity.max() > 0:
            scale = mixed_intensity.max() + 1e-8
            mixed_intensity = mixed_intensity / scale
            base_intensities = [t / scale for t in base_intensities]
            
        # Padding to max_k
        while len(base_intensities) < self.max_k:
            base_intensities.append(np.zeros_like(mixed_intensity))
            base_ids.append(-1)
            base_weights.append(0.0)
            
        return {
            'multiphase_xrd': torch.tensor(mixed_intensity, dtype=torch.float32),
            'single_xrds': torch.tensor(np.stack(base_intensities), dtype=torch.float32),
            'phase_ids': torch.tensor(base_ids, dtype=torch.long),
            'weights': torch.tensor(base_weights, dtype=torch.float32),
            'k': k
        }

def create_rruff_dataset(rruff_db_path, min_k=2, max_k=4, k_weights=None, min_weight=0.1, weight_distribution="random", target_length=3500, split='all', num_folds=5, fold=0, seed=42, virtual_epoch_length=1000):
    return RRUFFOnlineMixingDataset(
        rruff_db_path=rruff_db_path,
        min_k=min_k,
        max_k=max_k,
        k_weights=k_weights,
        min_weight=min_weight,
        weight_distribution=weight_distribution,
        target_length=target_length,
        split=split,
        num_folds=num_folds,
        fold=fold,
        seed=seed,
        virtual_epoch_length=virtual_epoch_length,
    )


def create_rruff_dataloader(rruff_db_path, batch_size=32, min_k=2, max_k=4, k_weights=None, min_weight=0.1, weight_distribution="random", target_length=3500, split='all', num_folds=5, fold=0, seed=42, distributed=False, num_workers=4, pin_memory=True, virtual_epoch_length=1000, dataset=None):
    if dataset is None:
        dataset = create_rruff_dataset(
            rruff_db_path=rruff_db_path,
            min_k=min_k,
            max_k=max_k,
            k_weights=k_weights,
            min_weight=min_weight,
            weight_distribution=weight_distribution,
            target_length=target_length,
            split=split,
            num_folds=num_folds,
            fold=fold,
            seed=seed,
            virtual_epoch_length=virtual_epoch_length,
        )
    
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=(split=='train'), drop_last=(split=='train'))
        shuffle = False 
    else:
        shuffle = (split == 'train')

    from torch.utils.data import DataLoader
    # Assuming XRDCollateFunction needs to be imported or is defined above
    # Need to make sure XRDCollateFunction is accessible
    from .core import XRDCollateFunction
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=XRDCollateFunction()
    )
