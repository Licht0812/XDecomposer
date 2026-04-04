import os
import sqlite3
import numpy as np
from ase.db import connect as ase_connect
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
import pickle

# =============================================================================
# Configuration
# =============================================================================

DB_PATH = "/data/group/project1/Crystal/UniqCryLabeled.db"
NPZ_DIR = "/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data"
CIF_DIR = "Novel-Space/All_CIFs"
REF_DIR = "Novel-Space/References"
MAPPING_OUT = "id_to_ref_mapping_full.pkl"
VALID_IDS_OUT = "valid_crystal_ids.pkl"

os.makedirs(CIF_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

def get_cif_filename(struct):
    """
    Generate formula_spacegroup.cif filename convention.
    """
    formula = struct.composition.reduced_formula
    try:
        # Get international space group symbol and replace '/' with '-'
        sg_symbol = struct.get_space_group_info()[0].replace('/', '-')
    except:
        sg_symbol = "None"
    return f"{formula}_{sg_symbol}.cif"

def is_valid_npz(fpath):
    """Check if npz file exists and contains valid data."""
    if not os.path.exists(fpath):
        return False
    try:
        data = np.load(fpath)
        y = data['y'] if 'y' in data else data['intensity']
        if y is None or len(y) == 0 or np.max(y) == 0:
            return False
        return True
    except:
        return False

def main():
    db = ase_connect(DB_PATH)
    total_entries = db.count()
    print(f"Total entries in database: {total_entries}")

    id_to_ref = {}
    valid_ids = []
    
    # We'll use a dictionary to cache formula_sg to avoid redundant CIF writes
    cif_cache = {}

    print("Processing database entries...")
    # Use all entries
    for row in tqdm(db.select(), total=total_entries):
        try:
            # row.id is the SQLite autoincrement ID (1-based)
            # Label (PhaseID) is row.id - 1 (0-based)
            label = row.id - 1
            
            # Check for valid NPZ sample using row.id in filename
            has_valid_sample = False
            for s_idx in range(20):
                # Filename uses crystal_{row.id}_sample_{num}.npz
                npz_path = os.path.join(NPZ_DIR, f"crystal_{row.id}_sample_{s_idx:02d}.npz")
                if is_valid_npz(npz_path):
                    has_valid_sample = True
                    break
            
            if not has_valid_sample:
                continue

            # 2. Get Structure and generate CIF
            atoms = row.toatoms()
            struct = Structure(
                lattice=atoms.get_cell(),
                species=atoms.get_chemical_symbols(),
                coords=atoms.get_scaled_positions()
            )
            
            cif_name = get_cif_filename(struct)
            cif_path = os.path.join(CIF_DIR, cif_name)
            
            # Write CIF if it doesn't exist
            if cif_name not in cif_cache:
                if not os.path.exists(cif_path):
                    writer = CifWriter(struct)
                    writer.write_file(cif_path)
                cif_cache[cif_name] = True
            
            # 3. Store mapping (Label -> CIF name)
            id_to_ref[label] = cif_name
            valid_ids.append(label)

        except Exception as e:
            # print(f"Error processing row {row.id}: {e}")
            continue

    print(f"\nProcessed {len(valid_ids)} valid crystal IDs.")
    
    # Save mappings
    with open(MAPPING_OUT, 'wb') as f:
        pickle.dump(id_to_ref, f)
    with open(VALID_IDS_OUT, 'wb') as f:
        pickle.dump(valid_ids, f)
        
    print(f"Mapping saved to {MAPPING_OUT}")
    print(f"Valid IDs saved to {VALID_IDS_OUT}")

    # 4. Sync to References (autoXRD expects CIFs here)
    print(f"Syncing {len(cif_cache)} unique CIFs to {REF_DIR}...")
    for cif_name in cif_cache:
        src = os.path.join(CIF_DIR, cif_name)
        dst = os.path.join(REF_DIR, cif_name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    print("Rebuild complete.")

if __name__ == "__main__":
    main()
