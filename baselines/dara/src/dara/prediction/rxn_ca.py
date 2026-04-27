"""An interface for predicting products in a chemical reaction using a cellular
automaton model.
"""

from __future__ import annotations

import logging
import sys
import typing

from dara.prediction.base import PredictionEngine

if typing.TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedStructureEntry

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class CellularAutomatonEngine(PredictionEngine):
    """TODO: Engine for predicting products in a chemical reaction."""

    def predict(
        precursors: list[str],
        temp: float,
        computed_entries: list[ComputedStructureEntry] | None = None,
        open_elem: str | None = None,
        chempot: float = 0,
        e_hull_cutoff: float = 0.05,
    ):
        """
        Predicts the intermediates/products of a mixture of precursors.

        Args:
            precursors: List of precursor formulas (no stoichiometry required)
            temp: Temperature in Kelvin
            computed_entries: Optional list of ComputedStructureEntry objects, otherwise
                will download from Materials Project using your MP_API key (must be stored
                in environment variables as $MP_API_KEY)
            open_elem: Optional open element (e.g., "O" for oxygen). If "O_air" is provided,
                will automatically default to oxygen with appropriate chemical potential
                (0.21 atm at desired temp).
            chempot: Optional chemical potential, defaults to 0 (standard state at the
                desired temp)
            e_hull_cutoff: Energy above hull cutoff by which to filter entries (default: takes
                all entries with an E_hull <= 50 meV/atom.)
        """
