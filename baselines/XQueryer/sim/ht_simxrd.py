from Pysimxrd import generator
from ase.db import connect
from ase import Atoms
import random
import concurrent.futures
from tqdm import tqdm

def process_entry(_id, database, times):
    """Simulate diffraction patterns for one database entry."""
    atom_list = []
    _target = []
    _chem_form = []
    _latt_dis = []
    _inten = []

    try:
        entry = database.get(id=_id)
        label = _id + 1
        atoms = entry.toatoms()

        for sim in range(times):
            atom_list.append(atoms)
            _target.append(label)
            _chem_form.append(atoms.get_chemical_formula())

            # Configure simulation parameters.
            deformation = True
            grainsize = random.uniform(2, 20)
            orientation = [random.uniform(0., 0.4), random.uniform(0., 0.4)]
            thermo_vib = random.uniform(0.0, 0.3)
            lattice_extinction_ratio = 0.01
            lattice_torsion_ratio = 0.01
            background_order = 6
            background_ratio = 0.05
            mixture_noise_ratio = 0.02
            dis_detector2sample = 500
            half_height_slit_detector_H = 5
            half_height_sample_S = 2.5
            zero_shift = random.uniform(-1.5, 1.5)

            # Generate the simulated diffraction pattern
            x, y = generator.parser(
                database=database, entry_id=_id,
                deformation=deformation, grainsize=grainsize, prefect_orientation=orientation,
                thermo_vibration=thermo_vib, lattice_extinction_ratio=lattice_extinction_ratio,
                lattice_torsion_ratio=lattice_torsion_ratio, background_order=background_order,
                background_ratio=background_ratio, mixture_noise_ratio=mixture_noise_ratio,
                dis_detector2sample=dis_detector2sample, half_height_slit_detector=half_height_slit_detector_H,
                half_height_sample =half_height_sample_S, zero_shift=zero_shift
            )

            _latt_dis.append(str(x))
            _inten.append(str(y))

    except Exception as e:
        print(f"An error occurred: crystal id = {_id}, error: {e}")
        return None

    return  atom_list, _target, _chem_form, _latt_dis, _inten

def simulator(db_file, sv_file, times=10):
    """Simulate patterns for all entries and save them to a database."""
    database = connect(db_file)
    total_entries = database.count()

    entries = list(range(1, total_entries + 1))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_entry, _id, database, times) for _id in entries]

        atom_lists, target_lists, chem_form_lists, latt_dis_lists, inten_lists = [], [], [], [],[]
        for future in tqdm(concurrent.futures.as_completed(results), total=len(entries), desc='Processing Entries'):
            result = future.result()
            if result is not None:
                atom_list, target, chem_form, latt_dis, inten = result
                atom_lists.extend(atom_list)
                target_lists.extend(target)
                chem_form_lists.extend(chem_form)
                latt_dis_lists.extend(latt_dis)
                inten_lists.extend(inten)

    databs = connect(sv_file)
    for k in tqdm(range(len(target_lists)), desc='Writing to Database'):
        id = k  # Pay attention
        try:
            atoms = Atoms(atom_lists[k])
            databs.write(atoms=atoms, latt_dis=latt_dis_lists[k], intensity=inten_lists[k], Label=target_lists[k])
        except Exception as e:
            print("An error occurred while writing to the database: ", e)
    return True

if __name__ == '__main__':
    db_file = './demo_mp.db'
    sv_file = './train.db'
    simulator(db_file, sv_file, times=1)
