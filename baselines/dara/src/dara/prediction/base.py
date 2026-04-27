"""Base classes for phase prediction engines."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from monty.json import MSONable


class PredictionEngine(MSONable, metaclass=ABCMeta):
    """Base definition for a reaction network."""

    def __init__(self):
        """Initialize the engine."""

    @abstractmethod
    def predict(self, reactants, temperature, open_elem, chempot, e_hull_cutoff) -> dict[str, float]:
        """Predict and rank the probability of appearance of products of a chemical
        reaction.
        """
