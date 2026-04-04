import sqlite3
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from pymatgen.core import Structure, Lattice

# --- Configuration ---
DB_PATH = "/data/group/project1/Crystal/UniqCryLabeled.db"
OUTPUT_DIR = "/data/home/zdhs0019/Projects/xrd_baselines/dara/dataset/uniqcry_cifs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_blob(blob, dtype=np.float64):
    """Decode ASE blob to numpy array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=dtype)

def generate_cifs():
    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found at {DB_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get table name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_name = cursor.fetchone()[0]
    
    # Get all structures
    # ASE stores data in columns: id, numbers, positions, cell
    cursor.execute(f"SELECT id, numbers, positions, cell FROM {table_name}")
    rows = cursor.fetchall()
    
    logger.info(f"Generating CIFs for {len(rows)} crystals from table '{table_name}'...")
    
    count = 0
    for row in tqdm(rows):
        crystal_id, numbers_blob, positions_blob, cell_blob = row
        
        cif_path = Path(OUTPUT_DIR) / f"{crystal_id}.cif"
        if cif_path.exists():
            continue
            
        try:
            # Decode blobs
            # Numbers are int32, positions and cell are float64
            numbers = np.frombuffer(numbers_blob, dtype=np.int32)
            positions = np.frombuffer(positions_blob, dtype=np.float64).reshape(-1, 3)
            cell = np.frombuffer(cell_blob, dtype=np.float64).reshape(3, 3)
            
            if len(numbers) == 0:
                continue
                
            # Create pymatgen structure
            lattice = Lattice(cell)
            structure = Structure(lattice, numbers, positions, coords_are_cartesian=True)
            
            # Save to CIF
            structure.to(filename=str(cif_path), fmt="cif")
            count += 1
                
        except Exception as e:
            # Silence errors for speed, but count them if needed
            continue
            
    conn.close()
    logger.info(f"Successfully generated {count} CIF files in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_cifs()
