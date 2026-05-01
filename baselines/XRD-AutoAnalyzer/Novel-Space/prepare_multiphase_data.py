
import os
import numpy as np
import sqlite3
import random
from tqdm import tqdm
from scipy.interpolate import interp1d
import json

# Constants
SEED = 7
random.seed(SEED)
np.random.seed(SEED)

MIN_K = 2
MAX_K = 4
MIN_WEIGHT = 0.15
TARGET_LENGTH = 3500
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0

DB_PATH = 'data/UniqCryLabeled.db'
NPZ_DIR = 'data/mp20-xrd_data/data'
OUTPUT_DIR = 'Multiphase_Data'

def load_crystal_info():
    """Load crystal ID and their corresponding spectrum file paths from directory."""
    # Since the DB doesn't directly map to the npz files we found,
    # and the files are named crystal_{id}_sample_{idx}.npz,
    # we scan the directory to build the mapping.

    print("Scanning directory for npz files...")
    crystal_dict = {}
    all_files = os.listdir(NPZ_DIR)
    for f in all_files:
        if f.endswith('.npz') and f.startswith('crystal_'):
            try:
                # crystal_100000_sample_00.npz -> id=100000
                parts = f.split('_')
                cid = int(parts[1])
                if cid not in crystal_dict:
                    crystal_dict[cid] = []
                crystal_dict[cid].append(f)
            except:
                continue
    print(f"Found {len(crystal_dict)} unique crystal IDs.")
    return crystal_dict

def preprocess_spectrum(fpath):
    """Load and return spectrum. Already 3500 points in these files."""
    try:
        data = np.load(os.path.join(NPZ_DIR, fpath))
        y = data['y']

        # If it's not 3500, interpolate
        if len(y) != TARGET_LENGTH:
            x = data['x']
            f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
            new_x = np.linspace(MIN_ANGLE, MAX_ANGLE, TARGET_LENGTH)
            y = f(new_x)

        # Ensure non-negative
        y = np.maximum(y, 0)
        return y
    except Exception as e:
        print(f"Error loading {fpath}: {e}")
        return None

def generate_mixed_data(crystal_ids, crystal_dict, num_samples, split_name):
    """Generate multiphase samples for a given split."""
    os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)

    results_x = []
    results_y_labels = []
    results_y_weights = []

    for i in tqdm(range(num_samples), desc=f"Generating {split_name}"):
        k = random.randint(MIN_K, MAX_K)
        selected_ids = random.sample(crystal_ids, k)

        # Dirichlet sampling for weights with MIN_WEIGHT constraint
        weights = np.zeros(k)
        valid_weights = False
        while not valid_weights:
            weights = np.random.dirichlet([1.0] * k)
            if np.all(weights >= MIN_WEIGHT):
                valid_weights = True

        mixed_signal = np.zeros(TARGET_LENGTH)
        component_signals = []

        for idx, cid in enumerate(selected_ids):
            # Randomly pick one spectrum for this ID
            fpath = random.choice(crystal_dict[cid])
            sig = preprocess_spectrum(fpath)
            if sig is None: continue

            # Mix according to weight (no individual normalization, norm_method="none")
            mixed_signal += sig * weights[idx]
            component_signals.append(sig * weights[idx])

        if len(component_signals) < k: continue

        # Max Scaling for mixed and components (to maintain physical consistency)
        max_val = np.max(mixed_signal)
        if max_val > 0:
            mixed_signal /= max_val
            # All components also divided by the same max_val
            # component_signals = [s / max_val for s in component_signals]

        # Add Gaussian noise relative to max (which is now 1.0)
        noise = np.random.normal(0, 0.01, TARGET_LENGTH) # 1% noise
        mixed_signal += noise
        mixed_signal = np.maximum(mixed_signal, 0)

        results_x.append(mixed_signal)
        results_y_labels.append(selected_ids)
        results_y_weights.append(weights)

    # Save as npz
    save_path = os.path.join(OUTPUT_DIR, f"{split_name}_data.npz")
    np.savez(save_path,
             x=np.array(results_x),
             labels=np.array(results_y_labels, dtype=object),
             weights=np.array(results_y_weights, dtype=object))
    print(f"Saved {split_name} data to {save_path}")

def main():
    import pickle
    crystal_dict = load_crystal_info()
    all_ids = sorted(list(crystal_dict.keys()))
    random.shuffle(all_ids)

    # 8:1:1 Split
    n = len(all_ids)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids_raw = all_ids[val_end:]

    # Filter test_ids to only include those in our reference library mapping
    MAPPING_PATH = 'id_to_ref_mapping_full.pkl'
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'rb') as f:
            id_to_ref = pickle.load(f)
        test_ids = [cid for cid in test_ids_raw if cid in id_to_ref]
        print(f"Filtered test IDs: {len(test_ids)} / {len(test_ids_raw)} (in reference library)")
    else:
        print(f"Warning: {MAPPING_PATH} not found. Using raw test IDs.")
        test_ids = test_ids_raw

    print(f"IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Generate data
    generate_mixed_data(train_ids, crystal_dict, 10000, "train")
    generate_mixed_data(val_ids, crystal_dict, 1000, "val")
    generate_mixed_data(test_ids, crystal_dict, 1000, "test")

if __name__ == "__main__":
    main()
