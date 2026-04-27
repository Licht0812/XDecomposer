
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction import xrd
import os

ref_dir = 'Novel-Space/References'
f = 'IrC_187.cif'
struc = Structure.from_file(os.path.join(ref_dir, f))
calculator = xrd.XRDCalculator()
pattern = calculator.get_pattern(struc, two_theta_range=(10, 80))
print(f"File {f} peaks: {pattern.x[:5]}")
