
import os
import numpy as np
import sqlite3
from pymatgen.core import Structure, Lattice
from tqdm import tqdm
import pickle

# Constants
DB_PATH = '/data/group/project1/Crystal/UniqCryLabeled.db'
NPZ_DIR = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data'
REF_DIR = 'Novel-Space/References'
OUTPUT_TRAIN_DATA = 'NPZ_Training_Data.npy'
OUTPUT_MAPPING = 'id_to_ref_mapping_full.pkl'

def get_structure_info(cur, cid):
    cur.execute("SELECT numbers, positions, cell FROM systems WHERE id=?", (cid,))
    row = cur.fetchone()
    if row:
        try:
            numbers = np.frombuffer(row[0], dtype=np.int32)
            positions = np.frombuffer(row[1], dtype=np.float64).reshape(-1, 3)
            cell = np.frombuffer(row[2], dtype=np.float64).reshape(3, 3)
            lattice = Lattice(cell)
            struc = Structure(lattice, numbers, positions)
            formula = struc.composition.reduced_formula
            sg = struc.get_space_group_info()[1]
            return f"{formula}_{sg}.cif"
        except:
            return None
    return None

def main():
    # 1. Get list of required classes (sorted to match autoXRD convention)
    ref_cifs = sorted([f for f in os.listdir(REF_DIR) if f.endswith('.cif')])
    ref_set = set(ref_cifs)
    print(f"Target classes: {len(ref_cifs)}")

    # 2. Scan NPZ directory to find available IDs
    print("Scanning NPZ directory...")
    available_ids = set()
    for f in os.listdir(NPZ_DIR):
        if f.startswith('crystal_') and f.endswith('_sample_00.npz'):
            try:
                cid = int(f.split('_')[1])
                available_ids.add(cid)
            except:
                continue
    print(f"Found {len(available_ids)} unique crystal IDs with NPZ files.")

    # 3. Map IDs to Reference CIFs
    print("Mapping IDs to References (this may take a while)...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    ref_to_ids = {cif: [] for cif in ref_cifs} # Fixed: Map class -> list of IDs
    id_to_ref = {} # Map ID -> class
    
    # We only check IDs that have NPZ files
    for cid in tqdm(available_ids):
        ref_label = get_structure_info(cur, cid)
        if ref_label in ref_set:
            ref_to_ids[ref_label].append(cid)
            id_to_ref[cid] = ref_label
            
    conn.close()
    
    mapped_classes = sum(1 for ids in ref_to_ids.values() if len(ids) > 0)
    print(f"Mapped {mapped_classes} / {len(ref_cifs)} classes.")
    print(f"Total mapped IDs: {len(id_to_ref)}")

    # Save mapping for evaluation
    with open(OUTPUT_MAPPING, 'wb') as f:
        pickle.dump(id_to_ref, f)

    # 4. Extract Spectra for Training
    print("Extracting spectra...")
    final_training_data = []
    
    for cif in tqdm(ref_cifs):
        class_spectra = []
        cids = ref_to_ids.get(cif, [])
        
        if not cids:
            class_spectra = [np.zeros(4501)] 
        else:
            samples_collected = 0
            for cid in cids:
                if samples_collected >= 20: break
                for s_idx in range(20):
                    if samples_collected >= 20: break
                    fname = f"crystal_{cid}_sample_{s_idx:02d}.npz"
                    fpath = os.path.join(NPZ_DIR, fname)
                    if os.path.exists(fpath):
                        data = np.load(fpath)
                        y = data['y']
                        if len(y) != 4501:
                            from scipy.interpolate import interp1d
                            x_old = np.linspace(10, 80, len(y))
                            x_new = np.linspace(10, 80, 4501)
                            f = interp1d(x_old, y, kind='cubic', fill_value="extrapolate")
                            y = f(x_new)
                        class_spectra.append(y)
                        samples_collected += 1
            
            if not class_spectra:
                class_spectra = [np.zeros(4501)]

        final_training_data.append(class_spectra)

    # 5. Save Training Data
    np.save(OUTPUT_TRAIN_DATA, np.array(final_training_data, dtype=object))
    print(f"Training data saved to {OUTPUT_TRAIN_DATA}")

if __name__ == "__main__":
    main()
