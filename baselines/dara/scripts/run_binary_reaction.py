import pickle
import re
from pathlib import Path

from pymatgen.core import Composition

from dara.search.core import search_phases
from dara.structure_db import ICSDDatabase

dataset_path = Path(__file__).parent.parent / "dataset" / "binary_reactions"
result_folder = Path(__file__).parent.parent / "benchmarks" / "binary_reactions"


def parse_chemical_system(file: Path) -> str:
    precursors = file.stem.split("_")[0].split("-")
    compositions = Composition({})
    for precursor in precursors:
        precursor = re.sub(r"^(\d+)", "", precursor)
        compositions += Composition(precursor)

    return compositions.chemical_system


if __name__ == "__main__":
    result_folder.mkdir(exist_ok=True, parents=True)

    for xrdml in dataset_path.glob("*.xrdml"):
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
