import ase.db
import numpy as np
import os

db_path = 'data/UniqCryLabeled.db'
npz_dir = 'data/UniqCry'

def check_db():
    print(f"Connecting to {db_path}...")
    db = ase.db.connect(db_path)

    labels = []
    ids = []
    for i, row in enumerate(db.select(limit=5000)):
        labels.append(int(getattr(row, 'Label')))
        ids.append(row.id)

    print(f"Total rows checked: {len(labels)}")
    print(f"Min Label: {min(labels)}, Max Label: {max(labels)}")
    print(f"Min row.id: {min(ids)}, Max row.id: {max(ids)}")

    # Check if Label is always row.id or row.id - 1
    diffs = [l - r for l, r in zip(labels, ids)]
    print(f"Unique (Label - row.id) values: {set(diffs)}")

def check_npz():
    print(f"\nChecking NPZ files in {npz_dir}...")
    files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')][:5]
    for f in files:
        path = os.path.join(npz_dir, f)
        with np.load(path) as data:
            keys = list(data.keys())
            print(f"File: {f}, Keys: {keys}")
            for k in keys:
                arr = data[k]
                if arr.ndim >= 1:
                    print(f"  Key: {k}, Shape: {arr.shape}, Max: {arr.max()}, Min: {arr.min()}, Mean: {arr.mean()}")

if __name__ == '__main__':
    check_db()
    check_npz()
