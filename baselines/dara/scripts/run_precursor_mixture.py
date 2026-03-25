import pickle
from pathlib import Path

from pymatgen.core import Composition

from dara.search.core import search_phases
from dara.structure_db import ICSDDatabase

dataset_path = Path(__file__).parent.parent / "dataset" / "precursor_mixture"
result_folder = Path(__file__).parent.parent / "benchmarks" / "precursor_mixture"
result_folder.mkdir(exist_ok=True, parents=True)

composition_mapping = {
    "Bi2O3-alpha": "Bi2O3",
    "Bi2O3-beta": "Bi2O3",
    "NaAlO2-1.5H2O": "NaAlO3.5H3",
}


def parse_chemical_system(file: Path) -> str:
    precursors = ".".join(file.stem.split(".")[:-1]).split("_")
    compositions = Composition({})
    for precursor in precursors:
        if precursor in composition_mapping:
            compositions += Composition(composition_mapping[precursor])
        else:
            compositions += Composition(precursor)

    return compositions.chemical_system


if __name__ == "__main__":
    for xrdml in dataset_path.glob("*.12-min.xrdml"):
        chemical_system = parse_chemical_system(xrdml)

        dest_folder = result_folder / xrdml.stem
        dest_folder.mkdir(exist_ok=True)
        cif_folder = dest_folder / "cifs"

        icsd = ICSDDatabase()
        icsd.get_cifs_by_chemsys(chemical_system, dest_dir=cif_folder.as_posix())

        search_tree = search_phases(
            pattern_path=xrdml,
            phases=list(cif_folder.glob("*.cif")),
            return_search_tree=True,
        )

        with open(dest_folder / "search_tree.pkl", "wb") as f:
            pickle.dump(search_tree, f)
