
import sqlite3
import numpy as np
from pymatgen.core import Structure, Lattice
import os
from tqdm import tqdm
import pickle

def get_structure_from_db(cur, cid):
    cur.execute("SELECT numbers, positions, cell FROM systems WHERE id=?", (cid,))
    row = cur.fetchone()
    if row:
        numbers = np.frombuffer(row[0], dtype=np.int32)
        positions = np.frombuffer(row[1], dtype=np.float64).reshape(-1, 3)
        cell = np.frombuffer(row[2], dtype=np.float64).reshape(3, 3)
        lattice = Lattice(cell)
        structure = Structure(lattice, numbers, positions)
        return structure
    return None

def build_mapping(db_path, ref_dir, sample_ids):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    id_to_label = {}
    print(f"Building mapping for {len(sample_ids)} unique IDs...")

    for cid in tqdm(sample_ids):
        struc = get_structure_from_db(cur, cid)
        if struc:
            formula = struc.composition.reduced_formula
            try:
                sg = struc.get_space_group_info()[1]
                label = f"{formula}_{sg}.cif"
                if os.path.exists(os.path.join(ref_dir, label)):
                    id_to_label[cid] = label
            except:
                pass
    conn.close()
    return id_to_label

# Load sample IDs from test data
test_data = np.load('Multiphase_Data/test_data.npz', allow_pickle=True)
labels = test_data['labels']
unique_ids = set()
for l in labels:
    for cid in l:
        unique_ids.add(cid)

db_path = 'data/UniqCryLabeled.db'
ref_dir = 'Novel-Space/References'
mapping = build_mapping(db_path, ref_dir, list(unique_ids))

print(f"Mapped {len(mapping)} out of {len(unique_ids)} IDs.")
if mapping:
    print(f"Example mapping: {list(mapping.items())[0]}")

with open('id_to_ref_mapping.pkl', 'wb') as f:
    pickle.dump(mapping, f)
