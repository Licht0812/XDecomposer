
import sqlite3
import numpy as np
from pymatgen.core import Structure, Lattice
import pickle
import json

def get_structure_from_db(db_path, cid):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT numbers, positions, cell FROM systems WHERE id=?", (cid,))
    row = cur.fetchone()
    conn.close()

    if row:
        numbers = np.frombuffer(row[0], dtype=np.int64)
        positions = np.frombuffer(row[1], dtype=np.float64).reshape(-1, 3)
        cell = np.frombuffer(row[2], dtype=np.float64).reshape(3, 3)

        lattice = Lattice(cell)
        structure = Structure(lattice, numbers, positions)
        return structure
    return None

db_path = 'data/UniqCryLabeled.db'
cid = 8604
struc = get_structure_from_db(db_path, cid)
if struc:
    print(f"ID {cid} Formula: {struc.composition.reduced_formula}")
    try:
        print(f"ID {cid} Space Group: {struc.get_space_group_info()[1]}")
    except:
        print(f"ID {cid} Space Group: None")
else:
    print(f"ID {cid} not found.")
