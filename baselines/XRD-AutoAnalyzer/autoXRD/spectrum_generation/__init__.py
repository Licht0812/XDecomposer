from autoXRD.spectrum_generation import strain_shifts, uniform_shifts, intensity_changes, peak_broadening, impurity_peaks, mixed
from multiprocessing import Pool, Manager
from pymatgen.core import Structure
from scipy import signal
import multiprocessing
import pymatgen as mg
import numpy as np
import math
import os

class SpectraGenerator(object):
    """
    Class used to generate augmented xrd spectra
    for all reference phases
    """

    def __init__(self, reference_dir, num_spectra=50, max_texture=0.6, min_domain_size=1.0, max_domain_size=100.0, max_strain=0.04, max_shift=0.25, impur_amt=70.0, min_angle=10.0, max_angle=80.0, separate=True, is_pdf=False, num_cpu=None):
        """
        Args:
            reference_dir: path to directory containing
                CIFs associated with the reference phases
        """
        if num_cpu is None:
            self.num_cpu = multiprocessing.cpu_count()
        else:
            self.num_cpu = num_cpu
        self.ref_dir = reference_dir
        self.num_spectra = num_spectra
        self.max_texture = max_texture
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_strain = max_strain
        self.max_shift = max_shift
        self.impur_amt = impur_amt
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.separate = separate
        self.is_pdf = is_pdf

    def augment(self, filename):
        """
        For a given phase, produce a list of augmented XRD spectra.
        By default, 50 spectra are generated per artifact, including
        peak shifts (strain), peak intensity change (texture), and
        peak broadening (small domain size).

        Args:
            filename: filename of the structure to be loaded.
        Returns:
            patterns: augmented XRD spectra
            filename: filename of the reference phase
        """

        struc = Structure.from_file(os.path.join(self.ref_dir, filename))
        patterns, pdf_specs = [], []

        if self.separate:
            patterns += strain_shifts.main(struc, self.num_spectra, self.max_strain, self.min_angle, self.max_angle)
            patterns += uniform_shifts.main(struc, self.num_spectra, self.max_shift, self.min_angle, self.max_angle)
            patterns += peak_broadening.main(struc, self.num_spectra, self.min_domain_size, self.max_domain_size, self.min_angle, self.max_angle)
            patterns += intensity_changes.main(struc, self.num_spectra, self.max_texture, self.min_angle, self.max_angle)
            patterns += impurity_peaks.main(struc, self.num_spectra, self.impur_amt, self.min_angle, self.max_angle)
        else:
            patterns += mixed.main(struc, 5*self.num_spectra, self.max_shift, self.max_strain, self.min_domain_size, self.max_domain_size,  self.max_texture, self.impur_amt, self.min_angle, self.max_angle)

        if self.is_pdf:
            for xrd in patterns:
                xrd = np.array(xrd).flatten()
                pdf = self.XRDtoPDF(xrd, self.min_angle, self.max_angle)
                pdf = [[v] for v in pdf]
                pdf_specs.append(pdf)
            return (pdf_specs, filename)

        return (patterns, filename)

    @property
    def augmented_spectra(self):

        filenames = sorted(os.listdir(self.ref_dir))
        total_phases = len(filenames)
        print(f"Generating augmented spectra for {total_phases} phases using {self.num_cpu} CPUs...")

        with Manager() as manager:
            pool = Pool(self.num_cpu)
            grouped_xrd = []
            # Optimization: Using a simpler loop and explicit deletion to help GC
            for i, result in enumerate(pool.imap(self.augment, filenames), 1):
                # result is (patterns, filename)
                # patterns is a list of augmented spectra (e.g. 1 or 5 spectra)
                # each spectrum is a list/array of 4501 values

                # Convert to float32 immediately to save 50% memory vs float64
                patterns = np.array(result[0], dtype=np.float32)
                grouped_xrd.append((patterns, result[1]))

                if i % 100 == 0 or i == total_phases:
                    print(f"  Progress: {i}/{total_phases} phases processed...", flush=True)

            print("Sorting and finalizing augmented spectra...")
            sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1]) ## Sort by filename

            # Extract only the patterns and convert to final numpy array
            # Use float32 to save memory
            sorted_spectra = np.array([group[0] for group in sorted_xrd], dtype=np.float32)

            # Explicitly clear intermediate list
            del grouped_xrd
            del sorted_xrd

            return sorted_spectra

    def XRDtoPDF(self, xrd, min_angle, max_angle):

        thetas = np.linspace(min_angle/2.0, max_angle/2.0, 4501)
        Q = np.array([4*math.pi*math.sin(math.radians(theta))/1.5406 for theta in thetas])
        S = np.array(xrd).flatten()

        pdf = []
        R = np.linspace(1, 40, 1000) # Only 1000 used to reduce compute time
        integrand = Q * S * np.sin(Q * R[:, np.newaxis])

        pdf = (2*np.trapz(integrand, Q) / math.pi)
        pdf = list(signal.resample(pdf, 4501))

        return pdf

