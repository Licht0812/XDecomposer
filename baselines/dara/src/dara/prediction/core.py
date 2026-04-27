"""Code for predicting products in a chemical reaction."""

from __future__ import annotations

import collections
import os
import shutil

from monty.json import MSONable
from rxn_network.utils.funcs import get_logger

from dara.structure_db import CODDatabase, StructureDatabase

logger = get_logger(__name__)


class PhasePredictor(MSONable):
    """Predict phases during solid-state synthesis."""

    def __init__(self, cif_dbs: list[StructureDatabase] | None = None, engine_name="reaction_network", **kwargs):
        """Initialize the engine."""
        if cif_dbs is None:
            cif_dbs = [CODDatabase()]

        self.cif_dbs = cif_dbs
        self.engine_name = engine_name
        self.engine = None

        if engine_name == "reaction_network":
            from dara.prediction.rn import ReactionNetworkEngine

            self.engine = ReactionNetworkEngine(**kwargs)

        if self.engine is None:
            raise ValueError(f"Unknown engine provided: {engine_name}")

    def predict(
        self,
        precursors: list[str],
        temp: float = 1000,
        computed_entries=None,
        open_elem=None,
        chempot: float = 0.0,
        e_hull_cutoff=0.05,
    ) -> dict[str, float]:
        """Predict and rank the probability of appearance of products of a chemical
        reaction.
        """
        if self.engine is None:
            raise ValueError("Engine not initialized!")

        return self.engine.predict(
            precursors=precursors,
            temp=temp,
            computed_entries=computed_entries,
            open_elem=open_elem,
            chempot=chempot,
            e_hull_cutoff=e_hull_cutoff,
        )

    def write_cifs_from_formulas(
        self,
        prediction: dict,
        cost_cutoff: float = 0.1,
        e_hull_filter: float = 0.05,
        dest_dir: str = "cifs",
        exclude_gases: bool = True,
    ):
        """Write CIFs of the predicted products."""
        prediction_sorted = collections.OrderedDict(sorted(prediction.items(), key=lambda item: item[1]))
        formulas = [formula for formula, cost in prediction_sorted.items() if cost < cost_cutoff]
        if os.path.exists(dest_dir):
            logger.info(f"Removing existing CIFs directory {dest_dir}")
            shutil.rmtree(dest_dir)

        for db in self.cif_dbs:
            db.get_cifs_by_formulas(
                formulas=formulas,
                e_hull_filter=e_hull_filter,
                copy_files=True,
                dest_dir=dest_dir,
                exclude_gases=exclude_gases,
            )
