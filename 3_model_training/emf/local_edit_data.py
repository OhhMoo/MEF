"""Exact chemistry-only alignment of fixed-size steering family graphs.

The nine-atom exploratory task uses the lead atom order as a shared coordinate
system. Canonical SMILES may reorder atoms after a mutation, so every candidate
is aligned by the minimum number of atom substitutions plus bond-label edits.
Neither response observations nor hidden campaign coefficients enter this map.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np

from .chem import decode_graph, smiles_to_graph
from .steering_data import SteeringDataset


DEFAULT_ALIGNMENT_PATH = Path("2_data_storage/processed/local_edit_steering/alignment.json")


def _adjacency(graph: dict) -> np.ndarray:
    count = int(graph["count"])
    maximum = len(graph["atoms"])
    matrix = np.zeros((maximum, maximum), dtype=np.uint8)
    upper = np.triu_indices(maximum, k=1)
    matrix[upper] = graph["bonds"]
    matrix += matrix.T
    return matrix[:count, :count]


def _label_distance(left: np.ndarray, right: np.ndarray, categories: int) -> int:
    """Minimum mismatches between two freely permutable equal-size label bags."""
    a = np.bincount(left, minlength=categories)
    b = np.bincount(right, minlength=categories)
    return int(np.abs(a - b).sum() // 2)


def _minimum_edit_permutation(lead: dict, candidate: dict) -> tuple[np.ndarray, int]:
    """Globally minimize atom plus bond edits, breaking ties by map tuple.

    Branch-and-bound lower bounds compare disjoint label bags: remaining atom
    labels, remaining internal bonds, and the bonds from each mapped atom to
    unmapped atoms. Each bag's mismatch count is a valid bound regardless of
    the final permutation, so pruning cannot remove an optimum.
    """
    n = int(lead["count"])
    if n != int(candidate["count"]):
        raise ValueError("local edit alignment requires equal atom counts")
    lead_atoms = np.asarray(lead["atoms"][:n], dtype=np.uint8)
    candidate_atoms = np.asarray(candidate["atoms"][:n], dtype=np.uint8)
    lead_bonds, candidate_bonds = _adjacency(lead), _adjacency(candidate)
    # High-degree atoms expose bond disagreements early in the search.
    order = tuple(sorted(range(n), key=lambda i: (-int(np.count_nonzero(candidate_bonds[i])), i)))
    mapping = [-1] * n  # candidate canonical index -> lead index
    best_map = tuple(range(n))

    def full_cost(permutation: tuple[int, ...]) -> int:
        atom_cost = sum(candidate_atoms[i] != lead_atoms[permutation[i]] for i in range(n))
        bond_cost = sum(candidate_bonds[i, j] != lead_bonds[permutation[i], permutation[j]]
                        for i in range(n) for j in range(i + 1, n))
        return int(atom_cost + bond_cost)

    best_cost = full_cost(best_map)

    def lower_bound(depth: int, used: int) -> int:
        remaining_candidate = order[depth:]
        remaining_lead = [i for i in range(n) if not (used >> i) & 1]
        bound = _label_distance(candidate_atoms[list(remaining_candidate)],
                                lead_atoms[remaining_lead], 4)
        if len(remaining_candidate) > 1:
            c = candidate_bonds[np.ix_(remaining_candidate, remaining_candidate)]
            l = lead_bonds[np.ix_(remaining_lead, remaining_lead)]
            upper = np.triu_indices(len(remaining_candidate), k=1)
            bound += _label_distance(c[upper], l[upper], 4)
        for prior in order[:depth]:
            bound += _label_distance(candidate_bonds[prior, list(remaining_candidate)],
                                     lead_bonds[mapping[prior], remaining_lead], 4)
        return bound

    def search(depth: int, used: int, cost: int) -> None:
        nonlocal best_cost, best_map
        if cost + lower_bound(depth, used) > best_cost:
            return
        if depth == n:
            result = tuple(mapping)
            if cost < best_cost or (cost == best_cost and result < best_map):
                best_cost, best_map = cost, result
            return
        atom = order[depth]
        choices = []
        for lead_index in range(n):
            if (used >> lead_index) & 1:
                continue
            step = int(candidate_atoms[atom] != lead_atoms[lead_index])
            step += sum(candidate_bonds[atom, prior] != lead_bonds[lead_index, mapping[prior]]
                        for prior in order[:depth])
            choices.append((int(step), lead_index))
        for step, lead_index in sorted(choices):
            if cost + step > best_cost:
                continue
            mapping[atom] = lead_index
            search(depth + 1, used | (1 << lead_index), cost + step)
            mapping[atom] = -1

    search(0, 0, 0)
    return np.asarray(best_map, dtype=np.int64), best_cost


def aligned_graph(lead_smiles: str, analogue_smiles: str, max_nodes: int = 9) -> dict:
    """Represent one same-count analogue in the canonical lead's atom order.

    ``canonical_to_lead[i]`` identifies the lead position corresponding to
    canonical candidate atom ``i``. The inverse ``lead_to_canonical`` reorders
    candidate tensors into lead coordinates. Edit counts compare every active
    atom and unordered pair after this exact minimum-edit alignment.
    """
    lead = smiles_to_graph(lead_smiles, max_nodes)
    candidate = smiles_to_graph(analogue_smiles, max_nodes)
    n = int(lead["count"])
    if n != int(candidate["count"]) or n != max_nodes:
        raise ValueError("first local edit study requires fixed full-size graphs")
    canonical_to_lead, minimum_cost = _minimum_edit_permutation(lead, candidate)
    lead_to_canonical = np.argsort(canonical_to_lead)
    atoms = np.asarray(candidate["atoms"], dtype=np.uint8)[lead_to_canonical].copy()
    matrix = _adjacency(candidate)
    aligned_matrix = matrix[np.ix_(lead_to_canonical, lead_to_canonical)]
    upper = np.triu_indices(max_nodes, k=1)
    bonds = aligned_matrix[upper].copy()
    atom_edits = int(np.count_nonzero(atoms != lead["atoms"]))
    bond_edits = int(np.count_nonzero(bonds != lead["bonds"]))
    assert atom_edits + bond_edits == minimum_cost
    decoded = decode_graph(atoms, bonds, n, max_nodes)
    if not decoded.chemically_valid or not decoded.connected or decoded.smiles != candidate["smiles"]:
        raise AssertionError("aligned graph failed molecular identity reconstruction")
    return {"smiles": candidate["smiles"], "atoms": atoms, "bonds": bonds,
            "count": n, "canonical_to_lead": canonical_to_lead,
            "lead_to_canonical": lead_to_canonical, "atom_edits": atom_edits,
            "bond_edits": bond_edits, "total_edits": minimum_cost}


def align_family_dataset(dataset, output_path=None) -> dict[int, dict[int, dict]]:
    """Cache all family members by ``cache[family_number][molecule_index]``.

    The cache uses only family membership, lead identity, and molecular graphs.
    It intentionally leaves source/validation/test NPZ observations untouched.
    """
    cache = {}
    for family_number, (lead_index, family_indices) in enumerate(
            zip(dataset.family_leads, dataset.family_indices)):
        lead_smiles = dataset.smiles[int(lead_index)]
        cache[family_number] = {
            int(index): aligned_graph(lead_smiles, dataset.smiles[int(index)], dataset.max_nodes)
            for index in family_indices}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "method": "exact_minimum_atom_plus_bond_edits",
                   "max_nodes": dataset.max_nodes,
                   "family_leads": [int(index) for index in dataset.family_leads],
                   "families": {str(family): {
                       str(index): {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                    for key, value in graph.items()}
                       for index, graph in records.items()}
                       for family, records in cache.items()}}
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    return cache


class AlignedSteeringDataset(SteeringDataset):
    """Existing steering episodes with lead-ordered query graphs only.

    Support graphs and the feature cache stay in canonical order because their
    molecular embeddings are invariant to atom numbering. Query graph labels
    are reordered to the episode lead before flow training. The same cache is
    reused for all response splits; its content depends on chemistry alone.
    """

    def __init__(self, data_dir, split="train", *, alignment_path=DEFAULT_ALIGNMENT_PATH):
        super().__init__(data_dir, split)
        path = Path(alignment_path)
        if not path.exists():
            align_family_dataset(self, path)
        payload = json.loads(path.read_text())
        if (payload.get("schema_version") != 1 or
                payload.get("method") != "exact_minimum_atom_plus_bond_edits" or
                payload.get("max_nodes") != self.max_nodes or
                payload.get("family_leads") != list(self.family_leads)):
            raise ValueError("alignment cache does not match steering chemistry")
        self.alignment_path = path
        self.alignment_cache = {}
        for family, indices in enumerate(self.family_indices):
            serialized = payload["families"][str(family)]
            if set(serialized) != {str(int(index)) for index in indices}:
                raise ValueError("alignment cache family membership differs")
            self.alignment_cache[family] = {}
            for index in indices:
                index = int(index)
                graph = serialized[str(index)]
                if graph["smiles"] != self.smiles[index] or graph["count"] != self.max_nodes:
                    raise ValueError("alignment cache identity differs")
                for field in ("atoms", "bonds", "canonical_to_lead", "lead_to_canonical"):
                    graph[field] = np.asarray(graph[field], dtype=np.uint8)
                self.alignment_cache[family][index] = graph

    def aligned_graph(self, family: int, index: int) -> dict:
        graph = self.alignment_cache[int(family)][int(index)]
        return {**graph, "id": self.ids[int(index)]}

    def sample_episode(self, rng, *, support_size=None, queries_per_lead=4,
                       campaign_index=None, lead_index=None):
        episode = super().sample_episode(
            rng, support_size=support_size, queries_per_lead=queries_per_lead,
            campaign_index=campaign_index, lead_index=lead_index)
        family = self.campaign_family[episode.campaign_id]
        queries = [self.aligned_graph(family, index) for index in episode.query_indices]
        return replace(episode, query_graphs=queries)
