"""An interface for rxn_network for predicting products in a chemical reaction."""

from __future__ import annotations

import collections
import itertools
import math
import typing

from maggma.stores.mongolike import MongoStore
from pymatgen.core import Composition, Element
from rxn_network.costs.calculators import (
    PrimaryCompetitionCalculator,
    SecondaryCompetitionCalculator,
)
from rxn_network.costs.functions import Softplus, WeightedSum
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.reactions.reaction_set import ReactionSet

from dara.prediction.base import PredictionEngine
from dara.utils import (
    get_chemsys_from_formulas,
    get_entries_in_chemsys_db,
    get_entries_in_chemsys_mp,
    get_logger,
)

if typing.TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedStructureEntry

logger = get_logger(__name__)


class ReactionNetworkEngine(PredictionEngine):
    """Engine for predicting products in a chemical reaction."""

    def __init__(self, cost_function="weighted_sum", max_rereact=10):
        """Initialize the engine.

        Args:
            cost_function: Cost function to use for ranking (default: weighted sum)
            max_rereact: Maximum number of phases to consider for followup reactions
                within the cost function
        """
        self.cost_function = cost_function
        self.max_rereact = max_rereact
        super().__init__()

    def predict(
        self,
        precursors: list[str],
        temp: float,
        computed_entries: list[ComputedStructureEntry] | MongoStore | None = None,
        open_elem: str | None = None,
        chempot: float = 0,
        e_hull_cutoff: float = 0.05,
    ):
        """
        Interface with reaction-network package for phase prediction based on interface
        reaction hulls.

        Args:
            precursors: List of precursor formulas (no stoichiometry required)
            temp: Temperature in Kelvin
            computed_entries: Optional list of ComputedStructureEntry objects or a store linking to an entry database.
                If None, will download from Materials Project using your MP_API key (must be stored in environment
                variables as $MP_API_KEY)
            open_elem: Optional open element (e.g., "O" for oxygen). If "O_air" is provided,
                will automatically default to oxygen with appropriate chemical potential
                (0.21 atm at desired temp).
            chempot: Optional chemical potential, defaults to 0 (standard state at the
                desired temp)
            e_hull_cutoff: Energy above hull cutoff by which to filter entries (default: takes
                all entries with an E_hull <= 50 meV/atom.)
        """
        precursors_comp = [Composition(p) for p in precursors]
        precursors_formulas = [p.reduced_formula for p in precursors_comp]
        temp = round(temp)
        cf = self._get_cost_function(temp)

        if open_elem == "O_air":
            open_elem = "O"
            chempot = (
                0.5 * 8.617e-5 * temp * math.log(0.21)
            )  # oxygen atmospheric partial pressure

        if computed_entries is None:
            logger.info("Downloading entries from Materials Project...")
            computed_entries = get_entries_in_chemsys_mp(
                get_chemsys_from_formulas(precursors_formulas)
            )
        elif isinstance(computed_entries, MongoStore):
            logger.info("Downloading entries from provided database...")
            computed_entries = get_entries_in_chemsys_db(
                computed_entries, get_chemsys_from_formulas(precursors_formulas)
            )
        else:
            logger.info("Using provided entries...")

        gibbs, precursors_no_open = self._get_entries(
            precursors_comp, computed_entries, open_elem, e_hull_cutoff, temp
        )

        logger.info("Enumerating all possible reactions...")
        rxns = self._enumerate_reactions(precursors_no_open, gibbs, open_elem, chempot)
        data = self._get_rxn_data(precursors_no_open, rxns, open_elem)
        ranked_formulas = self._rank_formulas(data, cf)

        rereact_formulas = list(ranked_formulas.keys())[: self.max_rereact]

        rereact_rxns = self._enumerate_reactions(
            rereact_formulas, gibbs, open_elem, chempot
        )
        rereact_data = self._get_rxn_data(rereact_formulas, rereact_rxns, open_elem)
        rereact_ranked_formulas = self._rank_formulas(rereact_data, cf)

        merged_dict = {
            key: min(
                ranked_formulas.get(key, float("inf")),
                rereact_ranked_formulas.get(key, float("inf")),
            )
            for key in set(ranked_formulas) | set(rereact_ranked_formulas)
        }
        for formula, cost in merged_dict.items():
            if formula in precursors_formulas:
                merged_dict[formula] = min(
                    cost, 0.0
                )  # make sure precursor is always there

        return collections.OrderedDict(
            sorted(merged_dict.items(), key=lambda item: item[1])
        )

    def _get_entries(
        self, precursors, computed_entries, open_elem, e_hull_cutoff, temp
    ):
        gibbs = GibbsEntrySet.from_computed_entries(
            computed_entries,
            300,
            include_nist_data=True,  # acquire and filter entries at 300 K
        )
        gibbs = gibbs.filter_by_stability(e_hull_cutoff)
        gibbs = gibbs.get_entries_with_new_temperature(temp)

        precursors_no_open_set = set(precursors)

        if open_elem:
            precursors_no_open_set = precursors_no_open_set - {
                Composition(
                    Composition(open_elem).reduced_formula
                )  # remove open element formula
            }

        precursors_no_open = list(precursors_no_open_set)

        reactants = [get_entry_by_formula(gibbs, r.reduced_formula) for r in precursors]

        gibbs_entries_list = gibbs.entries_list
        for entry in reactants:
            if entry not in gibbs_entries_list:
                gibbs_entries_list.append(entry)
        gibbs = GibbsEntrySet(gibbs_entries_list)

        return gibbs, precursors_no_open

    def _enumerate_reactions(self, precursors_no_open, gibbs, open_elem, chempot):
        rxns = BasicEnumerator(precursors=precursors_no_open).enumerate(gibbs)
        rxns = rxns.add_rxn_set(
            MinimizeGibbsEnumerator(precursors=precursors_no_open).enumerate(gibbs)
        )

        if open_elem:
            rxns = rxns.add_rxn_set(
                BasicOpenEnumerator(
                    [Composition(open_elem).reduced_formula],
                    precursors=precursors_no_open,
                ).enumerate(gibbs)
            )
            rxns = rxns.add_rxn_set(
                MinimizeGrandPotentialEnumerator(
                    open_elem=Element(open_elem),
                    mu=chempot,
                    precursors=precursors_no_open,
                ).enumerate(gibbs)
            )
            rxns = ReactionSet.from_rxns(
                rxns, rxns.entries, open_elem=open_elem, chempot=chempot
            )

        return rxns.filter_duplicates()

    def _get_rxn_data(self, precursors_no_open, rxns, open_elem):
        all_data = {}

        for combo in itertools.combinations(precursors_no_open, 2):
            combo = list(combo)
            if open_elem:
                combo.append(Composition(Composition(open_elem).reduced_formula))

            combo_rxns = list(rxns.get_rxns_by_reactants(combo))
            found_e1 = False
            found_e2 = False
            for rxn in combo_rxns:
                for e in rxn.reactant_entries:
                    if e.composition.reduced_composition == combo[0]:
                        found_e1 = True
                    elif e.composition.reduced_composition == combo[1]:
                        found_e2 = True
                    if found_e1 and found_e2:
                        break
            if not found_e1 or not found_e2:
                continue

            irh = InterfaceReactionHull(combo[0], combo[1], combo_rxns)

            for rxn in irh.reactions:
                calc1 = PrimaryCompetitionCalculator(irh)
                calc2 = SecondaryCompetitionCalculator(irh)

                rxn_decorated = calc1.decorate(rxn)
                rxn_decorated = calc2.decorate(rxn_decorated)

                for product in rxn_decorated.products:
                    if product not in all_data:
                        all_data[product] = [rxn_decorated]
                    else:
                        all_data[product].append(rxn_decorated)

        return all_data

    def _get_cost_function(self, temp):
        if self.cost_function == "weighted_sum":
            cf = WeightedSum(
                ["energy", "primary_competition", "secondary_competition"],
                [0.5, 0.25, 0.25],
            )
        elif self.cost_function == "softplus":
            cf = Softplus(
                temp=temp,
                params=["energy", "primary_competition", "secondary_competition"],
                weights=[0.5, 0.25, 0.25],
            )
        else:
            raise ValueError(f"Cost function {self.cost_function} not recognized.")

        return cf

    @staticmethod
    def _rank_formulas(data, cf):
        ranked_formulas = {}
        for comp, rxns in data.items():
            formula = comp.reduced_formula
            min_cost_rxn = min(rxns, key=cf.evaluate)
            min_cost = cf.evaluate(min_cost_rxn)
            ranked_formulas[formula] = min_cost

        return collections.OrderedDict(
            sorted(ranked_formulas.items(), key=lambda item: item[1])
        )

    @staticmethod
    def _get_probabilities(ranked_formulas):
        """Convert primary and secondary competition values into net probability of appearance."""
        phases, costs = ranked_formulas.keys(), ranked_formulas.values()
        inverse_rankings = [1 / (cost - min(costs) + 1) for cost in costs]
        total_inverse_rankings = sum(inverse_rankings)
        probabilities = [
            inv_rank / total_inverse_rankings for inv_rank in inverse_rankings
        ]
        return collections.OrderedDict(
            sorted(
                {
                    k.reduced_formula: round(v, 4)
                    for k, v in zip(phases, probabilities)
                }.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )


def get_entry_by_formula(gibbs_entries: GibbsEntrySet, formula: str):
    """Either returns the minimum energy entry or a new interpolated entry."""
    try:
        entry = gibbs_entries.get_min_entry_by_formula(formula)
    except:  # noqa: E722
        entry = gibbs_entries.get_interpolated_entry(
            formula
        )  # if entry is missing, use interpolated one
    return entry
