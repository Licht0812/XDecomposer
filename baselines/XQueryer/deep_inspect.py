import ase.db
import numpy as np
import os
import torch

db_path = 'data/UniqCryLabeled.db'
npz_dir = 'data/UniqCry'

def deep_inspect():
    print(f"--- Deep Inspecting DB: {db_path} ---")
    db = ase.db.connect(db_path)

    # Inspect the first 20 records
    print(f"{'Row ID':<10} | {'Label':<10} | {'Formula':<15} | {'Atoms Count':<10}")
    print("-" * 50)

    rows_data = []
    for i, row in enumerate(db.select(limit=20)):
        label = getattr(row, 'Label', 'N/A')
        formula = row.formula
        atoms_count = len(row.toatoms())
        print(f"{row.id:<10} | {label:<10} | {formula:<15} | {atoms_count:<10}")
        rows_data.append((row.id, label))

    # Check label uniqueness and nulls
    all_labels = [getattr(row, 'Label', None) for row in db.select()]
    unique_labels = set(all_labels)
    print(f"\nTotal rows: {len(all_labels)}")
    print(f"Unique Labels: {len(unique_labels)}")
    print(f"None Labels: {all_labels.count(None)}")
    if len(unique_labels) > 0:
        print(f"Min Label: {min([l for l in unique_labels if l is not None])}")
        print(f"Max Label: {max([l for l in unique_labels if l is not None])}")

    # Compare NPZ names with labels
    print(f"\n--- Checking NPZ files in {npz_dir} ---")
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f"Total NPZ files: {len(npz_files)}")

    sample_npz = npz_files[:5]
    for f in sample_npz:
        label_from_name = f.split('_')[1]
        path = os.path.join(npz_dir, f)
        with np.load(path) as data:
            # Check whether intensity is all zero
            for k in data.keys():
                arr = data[k]
                if arr.ndim >= 1:
                    print(f"File: {f} | Key: {k} | Shape: {arr.shape} | Max: {arr.max():.4f} | Sum: {arr.sum():.4f}")

if __name__ == '__main__':
    deep_inspect()
