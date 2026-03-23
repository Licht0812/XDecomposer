
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction import xrd
import sqlite3

def get_structure_from_db(db_path, cid):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT numbers, positions, cell FROM systems WHERE id=?", (cid,))
    row = cur.fetchone()
    conn.close()
    if row:
        numbers = np.frombuffer(row[0], dtype=np.int32)
        positions = np.frombuffer(row[1], dtype=np.float64).reshape(-1, 3)
        cell = np.frombuffer(row[2], dtype=np.float64).reshape(3, 3)
        lattice = Lattice(cell)
        structure = Structure(lattice, numbers, positions)
        return structure
    return None

db_path = '/data/group/project1/Crystal/UniqCryLabeled.db'
cid = 1
struc = get_structure_from_db(db_path, cid)

# Simulate peaks
calculator = xrd.XRDCalculator()
pattern = calculator.get_pattern(struc, two_theta_range=(10, 80))
sim_peaks = pattern.x

# Load NPZ peaks
npz_path = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data/crystal_1_sample_00.npz'
data = np.load(npz_path)
x_npz = data['x']
y_npz = data['y']
from scipy.signal import find_peaks
npz_peak_indices, _ = find_peaks(y_npz, height=5)
npz_peaks = x_npz[npz_peak_indices]

print(f"Simulated peaks (first 5): {sim_peaks[:5]}")
print(f"NPZ peaks (first 5): {npz_peaks[:5]}")
