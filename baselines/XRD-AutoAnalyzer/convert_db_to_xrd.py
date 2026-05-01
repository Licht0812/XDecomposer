import os
import numpy as np
import argparse
from tqdm import tqdm
from ase.db import connect as ase_connect
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

"""Export CIF and XY files with formula_spacegroup filenames."""

def process_db(db_path, xrd_dir, output_dir, limit=None):
    """
    Process UniqCryLabeled.db and associated .npz files using pymatgen
    to generate detailed CIF and .xy files.
    """
    # Create output directories
    cif_dir = os.path.join(output_dir, 'References')
    xy_dir = os.path.join(output_dir, 'Spectra')
    os.makedirs(cif_dir, exist_ok=True)
    os.makedirs(xy_dir, exist_ok=True)

    # Connect to ASE database to read the entries
    db = ase_connect(db_path)

    # Get total count for tqdm
    total = limit if limit else db.count()

    print(f"Starting processing {total} entries with pymatgen...")

    count = 0
    for row in tqdm(db.select(limit=limit), total=total):
        try:
            # Convert the ASE row to a pymatgen structure.
            atoms = row.toatoms()

            # Create pymatgen Structure
            struct = Structure(
                lattice=atoms.get_cell(),
                species=atoms.get_chemical_symbols(),
                coords=atoms.get_scaled_positions()
            )

            # Get space group information
            try:
                sg_symbol = struct.get_space_group_info()[0].replace('/', '-') # Avoid '/' in filenames
            except:
                sg_symbol = "None"

            # Metadata from DB
            kvp = row.key_value_pairs
            mpid = kvp.get('mpid', f"crystal_{row.id}").replace('.cif', '')
            label = kvp.get('Label', 'unknown')

            # Write the CIF with a formula_spacegroup filename.
            formula = struct.composition.reduced_formula
            cif_filename = f"{formula}_{sg_symbol}.cif"
            cif_path = os.path.join(cif_dir, cif_filename)

            writer = CifWriter(struct)
            writer.write_file(cif_path)

            # Locate the matching XRD npz file.
            npz_candidates = [
                os.path.join(xrd_dir, f"{row.id}.npz"),
                os.path.join(xrd_dir, f"crystal_{row.id}.npz"),
                os.path.join(xrd_dir, f"{mpid}.npz")
            ]

            npz_path = None
            for p in npz_candidates:
                if os.path.exists(p):
                    npz_path = p
                    break

            if npz_path:
                data = np.load(npz_path, allow_pickle=True)
                if 'x' in data and 'y' in data:
                    x = data['x']
                    y = data['y']
                    xy_filename = f"{formula}_{sg_symbol}.xy"
                    xy_path = os.path.join(xy_dir, xy_filename)
                    np.savetxt(xy_path, np.column_stack((x, y)), fmt='%.6f\t%.6f')

            count += 1
        except Exception as e:
            # print(f"Error processing row {row.id}: {e}")
            continue

    print(f"\nSuccessfully processed {count} entries.")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DB and NPZ to detailed CIF and XY using pymatgen")
    parser.add_argument("--db", default="data/UniqCryLabeled.db", help="Path to DB file")
    parser.add_argument("--xrd", default="data/xrd_data", help="Path to XRD npz directory")
    parser.add_argument("--out", default="./converted_data_pymatgen", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries to process")

    args = parser.parse_args()
    process_db(args.db, args.xrd, args.out, args.limit)
