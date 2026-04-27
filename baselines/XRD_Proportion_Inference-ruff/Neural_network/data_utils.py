
import sqlite3
import numpy as np
import torch
import torch.utils.data as data
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import os
import glob
from tqdm import tqdm

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares (ALS) for baseline correction.
    An implementation of Eilers's algorithm.
    (P. H. C. Eilers, Anal. Chem. 75, 3631 (2003))
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    y = np.asarray(y)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D.dot(D.transpose()) * lam
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

class RRUFFOnlineMixingDataset(data.Dataset):
    """
    Dataset for online mixing of single-phase XRD patterns from the RRUFF database.
    
    This dataset dynamically creates multiphase XRD patterns by mixing single-phase
    patterns on-the-fly. It handles data loading, preprocessing (baseline correction,
    interpolation, normalization), and k-fold splitting.
    """
    def __init__(self, rruff_db_path, min_k=2, max_k=4, k_weights=None, min_weight=0.1, 
                 weight_distribution="random", target_length=3500, theta_min=10.0, 
                 theta_max=80.0, split='all', num_folds=5, fold=0, seed=42):
        """
        Args:
            rruff_db_path (str): Path to the RRUFF ASE database file (e.g., 'UniqRruffCrystal.db').
            min_k (int): Minimum number of phases to mix.
            max_k (int): Maximum number of phases to mix.
            k_weights (list, optional): Weights for choosing k. Defaults to None (uniform).
            min_weight (float): Minimum weight for each phase in a mixture.
            weight_distribution (str): Strategy for weight generation ('random' or 'equal').
            target_length (int): Number of points in the standardized XRD pattern.
            theta_min (float): Minimum 2-theta angle for the standardized pattern.
            theta_max (float): Maximum 2-theta angle for the standardized pattern.
            split (str): Data split ('all', 'train', 'val', or 'test').
            num_folds (int): Number of folds for cross-validation.
            fold (int): The current fold to use for 'val' or 'test' split.
            seed (int): Random seed for reproducible k-fold splits.
        """
        self.rruff_db_path = rruff_db_path
        self.min_k = min_k
        self.max_k = max_k
        self.k_weights = k_weights
        self.min_weight = min_weight
        self.weight_distribution = weight_distribution
        self.target_length = target_length
        self.target_angles = np.linspace(theta_min, theta_max, target_length)
        
        # Load all valid phase data into memory
        self.phases = []  # List of (id, rruff_id, intensity_array)
        
        # Fast load from preprocessed directory if it exists
        preprocessed_dir = os.path.join(os.path.dirname(rruff_db_path), "rruff_processed")
        if os.path.isdir(preprocessed_dir):
            print(f"Loading preprocessed RRUFF npz files from {preprocessed_dir}...")
            self._load_preprocessed(preprocessed_dir)
        else:
            print("Processing from standard Database (This will take a while for ALS baseline removal)...")
            self._load_db()
            
        print(f"Loaded {len(self.phases)} total valid XRD patterns from RRUFF before splitting.")
        
        # Perform K-Fold Split if not using all data
        if split != 'all':
            self._perform_kfold_split(split, num_folds, fold, seed)
        
    def _load_preprocessed(self, prep_dir):
        """Loads pre-processed and cached .npz files."""
        npz_files = glob.glob(os.path.join(prep_dir, "rruff_*.npz"))
        for f in tqdm(npz_files, desc="Loading Cached NPZ"):
            try:
                fname = os.path.basename(f)
                s_id = int(fname.split("_")[1].split(".")[0])
                data = np.load(f)
                self.phases.append((s_id, f"ID_{s_id}", data['y'].astype(np.float32)))
            except Exception as e:
                print(f"Warning: Could not load {f}. Skipping. Error: {e}")

    def _load_db(self):
        """Loads data from the ASE database, performs preprocessing, and caches results."""
        from ase.db import connect
        db = connect(self.rruff_db_path)
        
        # Prepare cache directory
        preprocessed_dir = os.path.join(os.path.dirname(self.rruff_db_path), "rruff_processed")
        os.makedirs(preprocessed_dir, exist_ok=True)

        for row in tqdm(db.select(), desc="Processing ASE DB", total=db.count()):
            s_id = row.id
            data_dict = row.data
            if not data_dict or "angle" not in data_dict or "intensity" not in data_dict:
                continue
            
            try:
                angles = np.array(data_dict["angle"])
                intensities = np.array(data_dict["intensity"])
                
                # 1. Baseline Correction
                try:
                    baseline = baseline_als(intensities)
                    intensities = np.clip(intensities - baseline, 0, None)
                except Exception:
                    pass  # Ignore if ALS fails

                # 2. Interpolation to standard grid
                f = interp1d(angles, intensities, bounds_error=False, fill_value=0.0)
                std_intensities = f(self.target_angles)
                std_intensities = np.clip(std_intensities, 0, None)

                # 3. Normalization
                if std_intensities.max() > 0:
                    std_intensities = std_intensities / std_intensities.max()
                    
                rruff_id = row.get("rruff_id", f"ID_{s_id}")
                self.phases.append((s_id, rruff_id, std_intensities.astype(np.float32)))
                
                # 4. Cache the processed pattern
                np.savez_compressed(os.path.join(preprocessed_dir, f"rruff_{s_id}.npz"), y=std_intensities)

            except Exception as e:
                print(f"Error processing entry {s_id}: {e}")

    def _perform_kfold_split(self, split, num_folds, fold, seed):
        """Splits the data into train/validation sets for k-fold cross-validation."""
        # Sort phases by ID to ensure determinism
        self.phases.sort(key=lambda x: x[0])
        
        rng = random.Random(seed)
        indices = list(range(len(self.phases)))
        rng.shuffle(indices)
        
        fold_size = len(indices) // num_folds
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < num_folds - 1 else len(indices)
        val_indices = set(indices[val_start:val_end])
        
        if split == 'train':
            self.phases = [self.phases[i] for i in range(len(self.phases)) if i not in val_indices]
        elif split in ['val', 'test']:
            self.phases = [self.phases[i] for i in range(len(self.phases)) if i in val_indices]
            
        print(f"[{split.upper()} Fold {fold+1}/{num_folds}] Using {len(self.phases)} base patterns for mixing.")

    def __len__(self):
        # A "virtual" length for an epoch, since mixing is online and endless.
        # This value can be adjusted depending on training needs.
        return 10000

    def __getitem__(self, idx):
        """Generates one online-mixed sample."""
        # 1. Determine number of phases (k) for this sample
        if self.k_weights:
            k = random.choices(range(self.min_k, self.max_k + 1), weights=self.k_weights, k=1)[0]
        else:
            k = random.randint(self.min_k, self.max_k)
            
        # 2. Sample k phases from the available pool
        chosen_phases = random.sample(self.phases, k)
        
        # 3. Generate weights for the mixture
        if self.weight_distribution == "equal":
            weights = np.ones(k) / k
        else:  # "random"
            # Ensure each component has at least min_weight
            rem = 1.0 - k * self.min_weight
            if rem < 0: # Fallback if min_weight is too high
                weights = np.ones(k) / k
            else:
                raw_weights = np.random.dirichlet(np.ones(k))
                weights = raw_weights * rem + self.min_weight
        
        # 4. Mix the patterns
        mixed_intensity = np.zeros(self.target_length, dtype=np.float32)
        base_intensities = []
        base_ids = []
        
        for i, (p_id, r_id, p_intensity) in enumerate(chosen_phases):
            weighted_intensity = weights[i] * p_intensity
            mixed_intensity += weighted_intensity
            base_intensities.append(weighted_intensity)
            base_ids.append(p_id)
            
        # 5. Normalize the final mixture and its components
        if mixed_intensity.max() > 0:
            scale = mixed_intensity.max()
            mixed_intensity /= scale
            base_intensities = [b / scale for b in base_intensities]
            
        # 6. Pad to max_k for consistent tensor shapes
        while len(base_intensities) < self.max_k:
            base_intensities.append(np.zeros_like(mixed_intensity))
            base_ids.append(-1) # Use -1 as a sentinel for non-existent phases
            weights = np.append(weights, 0.0)
            
        return {
            'multiphase_xrd': torch.from_numpy(mixed_intensity).unsqueeze(0),
            'single_xrds': torch.from_numpy(np.stack(base_intensities)),
            'phase_ids': torch.tensor(base_ids, dtype=torch.long),
            'weights': torch.tensor(weights, dtype=torch.float),
            'num_phases': k,
            'k': k
        }

def get_dataloaders(rruff_db_path, config, batch_size=32, num_workers=4, num_folds=5, fold=0):
    """Creates Train, Val, and Test dataloaders for K-Fold CV."""
    
    train_dataset = RRUFFOnlineMixingDataset(
        rruff_db_path, min_k=config.MIN_K, max_k=config.MAX_K, 
        min_weight=config.MIN_WEIGHT, target_length=config.XRD_LENGTH,
        split='train', num_folds=num_folds, fold=fold, seed=config.SEED
    )
    
    val_dataset = RRUFFOnlineMixingDataset(
        rruff_db_path, min_k=config.MIN_K, max_k=config.MAX_K, 
        min_weight=config.MIN_WEIGHT, target_length=config.XRD_LENGTH,
        split='val', num_folds=num_folds, fold=fold, seed=config.SEED
    )
    
    # test dataset is same as val in K-fold CV unless we have a separate holdout
    test_dataset = val_dataset
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# Configuration dummy to match previous code
class OnlineMixingConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

