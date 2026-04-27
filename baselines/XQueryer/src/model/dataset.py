import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ase.db
import json
import os
import random
import ast
import pickle
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class OnlineMixingConfig:
    """Configuration for Online Mixing XRD Dataset."""
    MIN_K: int = 2
    MAX_K: int = 4
    MIN_WEIGHT: float = 0.15
    K_DISTRIBUTION: str = "weighted"
    K_WEIGHTS: Tuple[float, ...] = (0.4, 0.3, 0.15, 0.1, 0.05)
    WEIGHT_DISTRIBUTION: str = "random"
    XRD_LENGTH: int = 3500
    NORM_METHOD: str = "max"
    AUGMENT: bool = True
    NOISE_LEVEL: float = 0.01
    SHIFT_TOLERANCE: int = 50
    SEED: int = 7

def load_npz_pattern(file_path: str, key_priority: List[str] = None, shift: int = 0) -> Optional[np.ndarray]:
    """Loads an XRD pattern from an .npz file safely with optional peak shift and quality check."""
    if not os.path.exists(file_path):
        return None
    if key_priority is None:
        key_priority = ['y', 'intensity', 'xrd_pattern', 'pattern', 'xrd']
    try:
        with np.load(file_path) as data:
            for key in key_priority:
                if key in data:
                    arr = data[key]
                    # Clean NaN/Inf values
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if arr.ndim == 2: arr = arr[0]
                    arr = arr.astype(np.float32)

                    # Data Quality Check: skip if all zero or max is too low
                    if np.max(np.abs(arr)) < 1e-6:
                        return None

                    # Apply peak shift data augmentation
                    if shift != 0:
                        if shift > 0:
                            arr = np.concatenate([np.zeros(shift), arr[:-shift]])
                        else:
                            arr = np.concatenate([arr[-shift:], np.zeros(-shift)])
                    
                    return arr
            return None
    except Exception:
        return None

def process_pattern(pattern: np.ndarray, length: int) -> torch.Tensor:
    """Standardizes pattern length."""
    tensor = torch.from_numpy(pattern).float()
    curr_len = tensor.shape[0]
    if curr_len > length: tensor = tensor[:length]
    elif curr_len < length:
        padding = torch.zeros(length - curr_len)
        tensor = torch.cat([tensor, padding])
    return tensor

class ASEDataset(Dataset):
    def __init__(self, db_path, npz_dir, mode='train', encode_element=True, num_classes=100315, 
                 config: Optional[OnlineMixingConfig] = None):
        
        # Get the directory of the current script (src/model/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        emb_path = os.path.join(current_dir, '..', 'CGCNN_atom_emb.json')
        
        with open(emb_path , 'r') as file:
            self.cgcnn_emb = json.load(file)
        
        self.db_path = db_path
        self.npz_dir = npz_dir
        self.mode = mode
        self.encode_element = encode_element
        self.num_classes = num_classes
        self.config = config if config is not None else OnlineMixingConfig()
        
        # Step 1: Build or Load Index
        self.label_to_data = self._build_or_load_index()
        all_labels = sorted(list(self.label_to_data.keys()))
        
        # Step 2: Split by ID
        random.seed(self.config.SEED)
        random.shuffle(all_labels)
        
        n = len(all_labels)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        if mode == 'train':
            self.active_labels = all_labels[:train_end]
        elif mode == 'val':
            self.active_labels = all_labels[train_end:val_end]
        else:
            self.active_labels = all_labels[val_end:]
            
        print(f"Loaded {mode} dataset: {len(self.active_labels)} phases (Crystal IDs)")

    def _build_or_load_index(self) -> Dict[int, Dict[str, Any]]:
        cache_path = os.path.join(os.path.dirname(self.db_path), "crystal_label_index.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f: return pickle.load(f)
            except: pass

        print(f"Indexing dataset (scanning {self.npz_dir})...")
        label_to_paths = {}
        for root, _, files in os.walk(self.npz_dir):
            for f in files:
                if f.endswith('.npz') and f.startswith('crystal_'):
                    try:
                        label = int(f.split('_')[1])
                        path = os.path.join(root, f)
                        if label not in label_to_paths: label_to_paths[label] = []
                        label_to_paths[label].append(path)
                    except: continue
        
        db = ase.db.connect(self.db_path)
        index = {}
        for row in db.select():
            label = int(getattr(row, 'Label'))
            # Fix potential ID mismatch: Ensure label alignment with multi_hot index
            # If the database row.id starts from 1 and is phase+1, we might need consistent indexing.
            # But the user says Label in filename corresponds to phase ID.
            if label in label_to_paths:
                atoms = row.toatoms()
                symbols = set(atoms.get_chemical_symbols())
                element_encode = self.symbol_to_atomic_number(symbols)
                element_values = [self.cgcnn_emb[str(code)] for code in element_encode if str(code) in self.cgcnn_emb]
                elem_emb = torch.mean(torch.tensor(element_values, dtype=torch.float32), dim=0) if element_values else torch.zeros(92)
                
                index[label] = {
                    'paths': label_to_paths[label],
                    'element_emb': elem_emb,
                    'symbols': symbols
                }
        
        try:
            with open(cache_path, 'wb') as f: pickle.dump(index, f)
        except: pass
        return index

    def symbol_to_atomic_number(self, symbol_list):
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
            'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        return [atomic_numbers.get(s, 0) for s in symbol_list]

    def __len__(self):
        # The number of available IDs (phases) is:
        # Train: ~80,000 | Val: ~10,000 | Test: ~10,000
        # Since we are doing "Online Mixing", one "sample" is a random mixture.
        # We can define how many mixtures constitute one "epoch".
        if self.mode == 'train':
            return 50000 # Larger epoch size to cover 80k IDs effectively
        elif self.mode == 'val':
            return 5000  # Sufficient samples for validation tracking
        else:
            return 10000 # Standard test set size

    def __getitem__(self, idx):
        # Mixing Logic aligned with reference/online_mixing.py
        if self.config.K_DISTRIBUTION == "weighted":
            k_options = list(range(self.config.MIN_K, self.config.MAX_K + 1))
            # Slice weights to match K range and re-normalize
            weights_subset = self.config.K_WEIGHTS[:len(k_options)]
            weights_subset = np.array(weights_subset) / sum(weights_subset)
            k = np.random.choice(k_options, p=weights_subset)
        else:
            k = random.randint(self.config.MIN_K, self.config.MAX_K)
            
        selected_labels = random.sample(self.active_labels, k)
        
        rem = 1.0 - k * self.config.MIN_WEIGHT
        weights = (np.random.dirichlet(np.ones(k)) * rem + self.config.MIN_WEIGHT) if rem > 0 else np.ones(k)/k
        
        mix = torch.zeros(self.config.XRD_LENGTH)
        multi_hot = torch.zeros(self.num_classes)
        ratios_vec = torch.zeros(self.num_classes)
        all_symbols = set()
        
        gt_xrds = torch.zeros(self.config.MAX_K, self.config.XRD_LENGTH)
        gt_ratios = torch.zeros(self.config.MAX_K)
        gt_ids = torch.full((self.config.MAX_K,), -1, dtype=torch.long)

        # Apply peak shift augmentation if enabled
        shift = 0
        if self.mode == 'train' and self.config.AUGMENT:
            shift = random.randint(-self.config.SHIFT_TOLERANCE, self.config.SHIFT_TOLERANCE)

        for i, label in enumerate(selected_labels):
            info = self.label_to_data[label]
            paths = info['paths']
            
            # Retry to find a valid non-zero pattern for this label
            t = None
            for _ in range(5): # Increase retries
                chosen_path = random.choice(paths)
                pattern = load_npz_pattern(chosen_path, shift=shift)
                if pattern is not None:
                    t = process_pattern(pattern, self.config.XRD_LENGTH)
                    break
            
            # If still None, try to pick a completely different label as a fallback
            # to ensure the mixture is always valid
            if t is None:
                max_fallback_attempts = 10
                for _ in range(max_fallback_attempts):
                    fallback_label = random.choice(self.active_labels)
                    if fallback_label in selected_labels: continue
                    fallback_paths = self.label_to_data[fallback_label]['paths']
                    pattern = load_npz_pattern(random.choice(fallback_paths), shift=shift)
                    if pattern is not None:
                        t = process_pattern(pattern, self.config.XRD_LENGTH)
                        label = fallback_label # Update label to the fallback one
                        info = self.label_to_data[label]
                        break
            
            if t is None:
                # Last resort: if still None (very unlikely), use zero but log it
                t = torch.zeros(self.config.XRD_LENGTH)
            
            mix += t * weights[i]
            multi_hot[int(label)] = 1.0
            ratios_vec[int(label)] = float(weights[i])
            all_symbols.update(info['symbols'])
            
            # Store ground truth for slot matching
            if i < self.config.MAX_K:
                gt_xrds[i] = t
                gt_ratios[i] = float(weights[i])
                gt_ids[i] = int(label)
            
        if self.mode == 'train' and self.config.AUGMENT:
            noise = torch.randn_like(mix) * self.config.NOISE_LEVEL * (mix.max() + 1e-8)
            mix = torch.clamp(mix + noise, min=0)
            
        scale = mix.max() + 1e-8 if self.config.NORM_METHOD == "max" else 1.0
        mix_norm = mix / scale

        # Scale GT XRDs as well? Usually we want to reconstruct the raw or scaled signal.
        # Let's scale GT XRDs by the same factor so they sum to mix_norm (approximately)
        gt_xrds = gt_xrds / scale

        if self.encode_element:
            element_encode = self.symbol_to_atomic_number(all_symbols)
            element_values = [self.cgcnn_emb[str(code)] for code in element_encode if str(code) in self.cgcnn_emb]
            element_value = torch.mean(torch.tensor(element_values, dtype=torch.float32), dim=0) if element_values else torch.zeros(92)
        else:
            element_value = torch.zeros(92)

        return {
            'intensity': mix_norm,
            'multi_hot': multi_hot,
            'ratios': ratios_vec,
            'element': element_value,
            'gt_xrds': gt_xrds,
            'gt_ratios': gt_ratios,
            'gt_ids': gt_ids
        }
    



class EXPDataset(Dataset):
    def __init__(self, db_paths,encode_element):
        with open('./CGCNN_atom_emb.json' , 'r') as file:
            self.cgcnn_emb = json.load(file)
        self.db_paths = db_paths
        self.encode_element = encode_element
        self.dbs = [ase.db.connect(db_path) for db_path in db_paths]
        print("Loaded data from:", db_paths)

    def __len__(self):
        total_length = sum(len(db) for db in self.dbs)
        return total_length

    def __getitem__(self, idx):
        
        cumulative_length = 0
        for i, db in enumerate(self.dbs):
            if idx < cumulative_length + len(db):
                # Adjust the index to the range of the current database
                adjusted_idx = idx - cumulative_length
                row = db.get(adjusted_idx + 1)  # EXP db indexing starts from 1
                if self.encode_element:
                    # In RRUFF database, the elements are saved in ATOM attribute
                    atoms = db.get_atoms(adjusted_idx + 1)
                    element = set(atoms.get_chemical_symbols())
                    element_encode = self.symbol_to_atomic_number(element)
                    element_value = []
                    for code in element_encode:
                        value = self.cgcnn_emb[str(code)]
                        element_value.append(value)
                    # mean pooling
                    element_value=torch.mean(torch.tensor(element_value, dtype=torch.float32),dim=0)
                # Extract relevant data from the row
         
                latt_dis = ast.literal_eval(getattr(row, 'angle'))
                intensity = ast.literal_eval(getattr(row, 'intensity'))

                """
                提前过滤数据,删除不对齐的情况
                min_length = min(len(latt_dis), len(intensity))
                latt_dis = latt_dis[:min_length]
                intensity = intensity[:min_length]
                """

                int_int = self.upsample(np.column_stack((latt_dis, intensity)))
                # the str ID of RRUFF database
                id_num = adjusted_idx +1 # adjusted_idx +1 is the real data index in RRUFF database
                
                # Convert to tensors
                #tensor_latt_dis = torch.tensor(latt_dis, dtype=torch.float32)
                tensor_intensity = torch.tensor(int_int, dtype=torch.float32)
                tensor_id = torch.tensor(id_num, dtype=torch.int64)
                if self.encode_element:
                    return {
                        #'latt_dis': tensor_latt_dis,
                        'intensity': tensor_intensity,
                        'id': tensor_id,
                        'element': element_value
                    }
                else:
                    return {
                        #'latt_dis': tensor_latt_dis,
                        'intensity': tensor_intensity,
                        'id': tensor_id,
                        'element': torch.zeros(92, dtype=torch.int)
                    }              
            cumulative_length += len(db)

    def symbol_to_atomic_number(self,symbol_list):
        # Mapping of element symbols to atomic numbers
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
            'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
            'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
            'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
            'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
            'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
            'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
            'Lv': 116, 'Ts': 117, 'Og': 118
        }
        
        atomic_number_list = []
        if symbol_list == []: atomic_number_list.append(0)
        else:
            for symbol in symbol_list:
                if symbol in atomic_numbers:
                    atomic_number_list.append(atomic_numbers[symbol])
                else:
                    atomic_number_list.append(0)  # Append None if symbol not in the dictionary
            
        return atomic_number_list
    
    def upsample(self, rows):
        rows = np.array(rows, dtype=object)
        _, unique_indices = np.unique(rows[:, 0], return_index=True)
        rows = rows[unique_indices]

        if float(rows[0][0]) > 10:
            rows = np.insert(rows, 0, ['10', float(rows[0][1])], axis=0)

        if float(rows[-1][0]) < 80:
            rows = np.append(rows, [['80', float(rows[-1][1])]], axis=0)

        rowsData = np.array(rows, dtype=np.float32)
        x = rowsData[:, 0].astype(np.float32)
        y = rowsData[:, 1].astype(np.float32)
        f = interp1d(x, y, kind='slinear', fill_value="extrapolate")
        xnew = np.linspace(10, 80, 3501)
        ynew = f(xnew)
        ynew = ynew / ynew.max() * 100

        return ynew
