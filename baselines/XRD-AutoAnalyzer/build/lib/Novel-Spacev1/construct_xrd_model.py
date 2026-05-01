from autoXRD import cnn, spectrum_generation, solid_solns, tabulate_cifs
import numpy as np
import os
import sys
import pymatgen as mg

if __name__ == '__main__':

    max_texture = 0.5 # default: texture associated with up to +/- 50% changes in peak intensities
    min_domain_size, max_domain_size = 5.0, 30.0 # default: domain sizes ranging from 5 to 30 nm
    max_strain = 0.03 # default: up to +/- 3% strain
    max_shift = 0.5 # default: up to +/- 0.5 degrees shift in two-theta
    impur_amt = 70.0 # Max amount of impurity phases to include (%)
    num_spectra = 50 # Number of spectra to simulate per phase
    separate = False # If False: apply all artifacts simultaneously
    min_angle, max_angle = 10.0, 80.0
    num_epochs = 50
    skip_filter = False
    include_elems = True
    enforce_order = False
    oxi_filter = False
    cif_dir = 'All_CIFs'
    ref_dir = 'References'
    for arg in sys.argv:
        if '--max_texture' in arg:
            max_texture = float(arg.split('=')[1])
        if '--min_domain_size' in arg:
            min_domain_size = float(arg.split('=')[1])
        if '--max_domain_size' in arg:
            max_domain_size = float(arg.split('=')[1])
        if '--max_strain' in arg:
            max_strain = float(arg.split('=')[1])
        if '--max_shift' in arg:
            max_shift = float(arg.split('=')[1])
        if '--impur_amt' in arg:
            impur_amt = float(arg.split('=')[1])
        if '--num_spectra' in arg:
            num_spectra = int(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--num_epochs' in arg:
            num_epochs = int(arg.split('=')[1])
        if '--skip_filter' in arg:
            skip_filter = True
        if '--ignore_elems' in arg:
            include_elems = False
        if '--enforce_order' in arg:
            enforce_order = True
        if '--oxi_filter' in arg:
            oxi_filter = True
        if '--separate_artifacts' in arg:
            separate = True
        if '--cif_dir=' in arg:
            cif_dir = arg.split('=')[1]
        if '--ref_dir=' in arg:
            ref_dir = arg.split('=')[1]

    if not skip_filter:
        assert os.path.isdir(cif_dir), 'No All_CIFs directory was provided. Please create or use --skip_filter'
        assert not os.path.exists(ref_dir), 'References directory already exists. Please remove or use --skip_filter'
        tabulate_cifs.main(cif_dir, ref_dir, oxi_filter, include_elems, enforce_order)

    else:
        assert os.path.isdir(ref_dir), '--skip_filter was specified, but no References directory was provided'

    if '--include_ns' in sys.argv:
        solid_solns.main(ref_dir)

    xrd_obj = spectrum_generation.SpectraGenerator(ref_dir, num_spectra, max_texture, min_domain_size,
        max_domain_size, max_strain, max_shift, impur_amt, min_angle, max_angle, separate, is_pdf=False)
    xrd_specs = xrd_obj.augmented_spectra

    if '--save' in sys.argv:
        np.save('XRD', np.array(xrd_specs))

    test_fraction = 0.2
    cnn.main(xrd_specs, num_epochs, test_fraction, is_pdf=False)
