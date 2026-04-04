import numpy as np
import os
import ase.db

db_path = '/data/group/project1/Crystal/UniqCryLabeled.db'
npz_dir = '/data/group/project1/Crystal/UniqCry'

# Scan files
id_to_paths = {}
for root, _, files in os.walk(npz_dir):
    for f in files:
        if f.endswith('.npz') and f.startswith('crystal_'):
            try:
                parts = f.split('_')
                crystal_id = int(parts[1])
                path = os.path.join(root, f)
                if crystal_id not in id_to_paths:
                    id_to_paths[crystal_id] = []
                id_to_paths[crystal_id].append(path)
            except: continue

db = ase.db.connect(db_path)
bad_labels = set()
total_labels = 0
checked_labels = 0

for row in db.select():
    label = getattr(row, 'Label')
    sys_id = row.id
    total_labels += 1
    
    if sys_id in id_to_paths:
        checked_labels += 1
        all_bad = True
        for path in id_to_paths[sys_id]:
            data = np.load(path)
            y = data['y']
            if not (np.isnan(y).any() or np.isinf(y).any()):
                all_bad = False
                break
        if all_bad:
            bad_labels.add(label)

print(f"Total rows in DB: {total_labels}")
print(f"Rows with matching files: {checked_labels}")
print(f"Labels where ALL files are bad: {len(bad_labels)}")
