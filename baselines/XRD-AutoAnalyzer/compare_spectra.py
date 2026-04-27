
import sqlite3
import numpy as np
from pymatgen.core import Structure, Lattice
from autoXRD import spectrum_analysis
import os

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
formula = struc.composition.reduced_formula
print(f"ID {cid} Formula: {formula}")

# Load NPZ spectrum
npz_path = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data/crystal_1_sample_00.npz'
data = np.load(npz_path)
x_npz = data['x']
y_npz = data['y']

# Simulate clean spectrum using autoXRD logic
from pymatgen.analysis.diffraction import xrd
calculator = xrd.XRDCalculator()
pattern = calculator.get_pattern(struc, two_theta_range=(10, 80))
x_sim = pattern.x
y_sim = pattern.y

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x_npz, y_npz/max(y_npz), label='NPZ (from user data)')
plt.stem(x_sim, y_sim/max(y_sim), linefmt='r-', markerfmt='ro', label='Simulated (clean)')
plt.legend()
plt.savefig('compare_spectra.png')
print("Comparison plot saved to compare_spectra.png")
