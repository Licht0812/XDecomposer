from autoXRD import cnn, spectrum_generation, solid_solns, tabulate_cifs
import numpy as np
import os
import sys
import pymatgen as mg

# Add TensorFlow environment variables to fix issues and force CPU if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if os.environ.get('CUDA_VISIBLE_DEVICES') == "":
    print("Forcing CPU-only mode as requested.")
else:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # Disable XLA JIT to avoid PTX version mismatch if driver is older than TF requirements
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'


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
    num_cpu = None
    for arg in sys.argv:
        if '--num_cpu' in arg:
            num_cpu = int(arg.split('=')[1])
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

        print(f"Directly generating References from {cif_dir} (O(N) Fast Mode)...", flush=True)
        from pymatgen.io.cif import CifParser
        import shutil

        if not os.path.isdir(ref_dir):
            os.mkdir(ref_dir)

        all_files = [f for f in os.listdir(cif_dir) if f.lower().endswith('.cif')]
        total_files = len(all_files)

        for i, f in enumerate(all_files):
            if i % 1000 == 0:
                print(f"  Processing: {i}/{total_files}...", flush=True)
            try:
                # Use occupancy_tolerance to be consistent with autoXRD
                parser = CifParser(os.path.join(cif_dir, f), occupancy_tolerance=1.25)
                struc = parser.parse_structures(primitive=False)[0]
                
                # Check for ordered structures if requested
                if enforce_order and not struc.is_ordered:
                    continue

                # Use tabulate_cifs.write_cifs to ensure naming consistency (Formula_SpaceGroup.cif)
                tabulate_cifs.write_cifs([struc], ref_dir, include_elems)
                
            except Exception:
                continue
        print(f"References generated in {ref_dir}. Total processed: {total_files}")

    else:
        assert os.path.isdir(ref_dir), '--skip_filter was specified, but no References directory was provided'

    if '--include_ns' in sys.argv:
        solid_solns.main(ref_dir)

    xrd_obj = spectrum_generation.SpectraGenerator(ref_dir, num_spectra, max_texture, min_domain_size,
        max_domain_size, max_strain, max_shift, impur_amt, min_angle, max_angle, separate, is_pdf=False, num_cpu=num_cpu)
    xrd_specs = xrd_obj.augmented_spectra

    if '--save' in sys.argv:
        np.save('XRD', np.array(xrd_specs))

    test_fraction = 0.2
    cnn.main(xrd_specs, num_epochs, test_fraction, is_pdf=False)
