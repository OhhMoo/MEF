"""Deterministic local chemistry and response-reversal episodes.

The public NPZ files contain measured outcomes, while the evaluator-only oracle
contains coefficient values. Family leads, mutation rules, descriptors, coefficient
splits, and goals are fixed in code before response feasibility is calculated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, rdMolDescriptors

from .campaigns import CONTEXT, EpisodeRecords, molecular_fingerprint
from .chem import decode_graph, smiles_to_graph
from .data import GraphDataset


SPLITS = ("train", "validation", "test")
SUPPORT_SIZES = (4, 16, 64)
GOAL = np.asarray([0.25, -0.10], dtype=np.float32)
MIN_SIMILARITY = 0.4
FAMILY_TARGET = 128
MAX_NODES = 9
# Ordered chemistry-only references. The order is also the declared replacement
# order if one reference cannot populate a family.
REFERENCE_SMILES = ("CC(C)CC(=O)NCC", "CC(=O)NCC(=O)CC")
DESCRIPTORS = {"u": "RDKit Crippen MolLogP / source-library std",
               "q": "RDKit topological polar surface area / source-library std"}
# Fresh final held-out values, declared after the discarded N/O construction
# pilot. None were used in that pilot's feasibility check.
VALIDATION_MAGNITUDES = (0.72, 1.52)
TEST_MAGNITUDES = (0.52, 0.92, 1.32, 1.92)
WITHHELD_HALF_WIDTH = 0.04


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _descriptors(smiles: str) -> np.ndarray:
    graph = smiles_to_graph(smiles, MAX_NODES)
    molecule = Chem.MolFromSmiles(graph["smiles"])
    return np.asarray([Crippen.MolLogP(molecule),
                       rdMolDescriptors.CalcTPSA(molecule)], dtype=np.float64)


def _same_count_mutations(graph: dict):
    """Same-count atom and bond edits from redesign_search.enumerate_mutations.

    Restricted here to its atom_substitution and bond_change branches. Edits
    retain a fixed graph size; decoding and the codec enforce valence and scope.
    """
    count = int(graph["count"])
    atoms, bonds = graph["atoms"], graph["bonds"]
    for atom in range(count):
        for label in range(4):
            if label != int(atoms[atom]):
                changed = atoms.copy()
                changed[atom] = label
                yield changed, bonds, count
    left, right = np.triu_indices(MAX_NODES, k=1)
    for index, j in enumerate(right):
        if j >= count:
            continue
        for label in range(4):
            if label != int(bonds[index]):
                changed = bonds.copy()
                changed[index] = label
                yield atoms, changed, count


def enumerate_family(reference: str, *, max_depth: int = 5) -> tuple[str, ...]:
    """Breadth-first, canonical, chemistry-only local mutation closure."""
    lead = smiles_to_graph(reference, MAX_NODES)
    reference = lead["smiles"]
    fingerprint = molecular_fingerprint(reference)
    seen = {reference}
    frontier = [reference]
    for _ in range(max_depth):
        next_frontier = set()
        for parent in frontier:
            for atoms, bonds, count in _same_count_mutations(smiles_to_graph(parent, MAX_NODES)):
                outcome = decode_graph(atoms, bonds, count, MAX_NODES)
                if not (outcome.chemically_valid and outcome.connected):
                    continue
                identity = outcome.smiles
                if identity in seen or identity in next_frontier:
                    continue
                try:
                    recoded = smiles_to_graph(identity, MAX_NODES)
                except ValueError:
                    continue
                if DataStructs.TanimotoSimilarity(
                        fingerprint, molecular_fingerprint(recoded["smiles"])) < MIN_SIMILARITY:
                    continue
                next_frontier.add(recoded["smiles"])
        frontier = sorted(next_frontier)
        seen.update(frontier)
        if len(seen) >= FAMILY_TARGET or not frontier:
            break
    return (reference, *sorted(seen - {reference}))


def _coefficients() -> dict[str, tuple[float, ...]]:
    # Six withheld magnitude neighborhoods are carved out before source draws.
    withheld = VALIDATION_MAGNITUDES + TEST_MAGNITUDES
    candidates = np.linspace(0.5, 2.0, 1001)
    available = candidates[np.all(np.abs(candidates[:, None] - np.asarray(withheld)) >
                                  WITHHELD_HALF_WIDTH, axis=1)]
    positions = np.linspace(0, len(available) - 1, 16).round().astype(int)
    train_magnitudes = tuple(float(available[index]) for index in positions)
    return {"train": train_magnitudes, "validation": VALIDATION_MAGNITUDES,
            "test": TEST_MAGNITUDES}


def paired_support_seed(campaign_id: str, *, base_seed: int = 142) -> int:
    """One outcome-independent reserve draw for both signs of a declared pair."""
    pair_key = campaign_id.rsplit("_", 1)[0]
    return base_seed + int.from_bytes(hashlib.sha256(pair_key.encode()).digest()[:4], "little")


@dataclass(frozen=True)
class SteeringOracle:
    """Evaluator-only exact centered response for supported novel molecules."""

    coefficients: dict[str, float]
    campaign_leads: dict[str, str]
    scales: tuple[float, float]

    def score_smiles(self, campaign_id: str, smiles: str) -> np.ndarray:
        if campaign_id not in self.coefficients:
            raise KeyError(campaign_id)
        candidate = _descriptors(smiles)
        lead = _descriptors(self.campaign_leads[campaign_id])
        delta = (candidate - lead) / np.asarray(self.scales)
        return np.asarray([self.coefficients[campaign_id] * delta[0], -delta[1]],
                          dtype=np.float64)

    evaluate_smiles = score_smiles

    @classmethod
    def load(cls, path) -> "SteeringOracle":
        values = json.loads(Path(path).read_text())
        if values["descriptors"] != DESCRIPTORS:
            raise ValueError("incompatible descriptor specification")
        return cls(values["coefficients"], values["campaign_leads"],
                   tuple(values["scales"]))


def prepare_steering_data(output_dir, *, seed: int = 142) -> dict:
    """Write one familiar graph library and independent coefficient splits.

    All references, coefficients, local members, and descriptors are determined
    without outcome feasibility. A complete per-case report follows preparation.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["seed"] != seed or
                manifest["descriptor_definition"] != DESCRIPTORS or
                manifest["reference_order"] != list(REFERENCE_SMILES) or
                manifest["validation_magnitudes"] != list(VALIDATION_MAGNITUDES) or
                manifest["test_magnitudes"] != list(TEST_MAGNITUDES)):
            raise FileExistsError("existing steering data uses another specification")
        if all(_digest(output_dir / entry["file"]) == entry["sha256"]
               for entry in manifest["splits"].values()):
            return manifest
        raise ValueError("steering cache integrity check failed")

    families = [enumerate_family(reference) for reference in REFERENCE_SMILES]
    if any(len(family) < FAMILY_TARGET for family in families):
        raise ValueError("chemical family below 128: " + str([len(x) for x in families]))
    identities = sorted(set().union(*(set(family) for family in families)))
    graphs = [dict(smiles_to_graph(identity, MAX_NODES), id=f"steer:{index:04d}")
              for index, identity in enumerate(identities)]
    index_of = {identity: index for index, identity in enumerate(identities)}
    family_indices = [[index_of[identity] for identity in family] for family in families]
    family_leads = [index_of[family[0]] for family in families]
    raw = np.stack([_descriptors(identity) for identity in identities])
    scales = np.std(raw, axis=0).clip(1e-6)
    coefficients_by_split = _coefficients()
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(exist_ok=True)
    coefficients, campaign_leads = {}, {}
    split_meta = {}
    observations_by_split = {}
    for split in SPLITS:
        campaign_ids, campaign_family, campaign_pairs = [], {}, {}
        observations = []
        for family_index, lead_index in enumerate(family_leads):
            lead_raw = raw[lead_index]
            for pair_index, magnitude in enumerate(coefficients_by_split[split]):
                plus = f"{split}_f{family_index}_p{pair_index:02d}_a"
                minus = f"{split}_f{family_index}_p{pair_index:02d}_b"
                for campaign_id, sign, opposite in ((plus, +1, minus), (minus, -1, plus)):
                    coefficient = sign * magnitude
                    coefficients[campaign_id] = coefficient
                    campaign_leads[campaign_id] = identities[lead_index]
                    campaign_ids.append(campaign_id)
                    campaign_family[campaign_id] = family_index
                    campaign_pairs[campaign_id] = opposite
                    delta = (raw - lead_raw) / scales
                    observations.append(np.column_stack((coefficient * delta[:, 0],
                                                          -delta[:, 1])).astype(np.float32))
        split_meta[split] = {"campaign_ids": campaign_ids, "campaign_family": campaign_family,
                             "campaign_pairs": campaign_pairs, "size": len(identities)}
        observations_by_split[split] = np.stack(observations)

    oracle_json = {"schema_version": 1, "descriptors": DESCRIPTORS,
                   "scales": scales.tolist(), "coefficients": coefficients,
                   "campaign_leads": campaign_leads,
                   "coefficient_magnitudes": {key: list(value)
                                              for key, value in coefficients_by_split.items()},
                   "withheld_half_width": WITHHELD_HALF_WIDTH}
    oracle_path = private_dir / "oracle.json"
    oracle_path.write_text(json.dumps(oracle_json, indent=2, sort_keys=True) + "\n")
    common = {"atoms": np.stack([graph["atoms"] for graph in graphs]),
              "bonds": np.stack([graph["bonds"] for graph in graphs]),
              "counts": np.asarray([graph["count"] for graph in graphs], dtype=np.uint8),
              "smiles": np.asarray(identities),
              "ids": np.asarray([graph["id"] for graph in graphs])}
    for split in SPLITS:
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **common, observations=observations_by_split[split],
                            campaign_ids=np.asarray(split_meta[split]["campaign_ids"]),
                            context=CONTEXT)
        split_meta[split].update(file=path.name, sha256=_digest(path))

    count_prior = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float64)
    for family in family_indices:
        count = int(graphs[family[0]]["count"])
        count_prior[count - 1, count - 1] = 1.0
    # This is a conditional proposal-count law. Both declared families have
    # nine atoms, so its supported row is deterministic.
    manifest = {"schema_version": 1, "dataset": "evidence_steering_toy",
                "seed": seed, "max_nodes": MAX_NODES, "support_sizes": list(SUPPORT_SIZES),
                "goal": GOAL.tolist(), "context": CONTEXT.tolist(),
                "reference_order": list(REFERENCE_SMILES),
                "family_leads": family_leads, "family_indices": family_indices,
                "family_sizes": list(map(len, families)),
                "family_membership_rule": "same heavy count; atom/bond edit BFS; canonical neutral C/N/O/F; Morgan2/2048 >= 0.4",
                "descriptor_definition": DESCRIPTORS,
                "descriptor_source_scales": scales.tolist(),
                "rdkit_version": rdBase.rdkitVersion,
                "coefficient_split_rule": "16 source magnitudes and both signs; fixed validation/test pairs; source excludes +/-0.04 around held-out magnitudes",
                "validation_magnitudes": list(VALIDATION_MAGNITUDES),
                "test_magnitudes": list(TEST_MAGNITUDES),
                "private_oracle_sha256": _digest(oracle_path),
                "count_prior": count_prior.tolist(), "splits": split_meta}
    pilot_path = output_dir / "construction_pilot.json"
    if pilot_path.exists():
        manifest["discarded_construction_pilot"] = {
            "file": pilot_path.name, "sha256": _digest(pilot_path)}
    # Fixed support pools are chosen without outcomes, then every case is
    # reported. The oracle metadata remains outside the public NPZ files.
    feasibility = []
    for split in SPLITS:
        for campaign_index, campaign_id in enumerate(split_meta[split]["campaign_ids"]):
            family = split_meta[split]["campaign_family"][campaign_id]
            lead = family_leads[family]
            members = np.asarray(family_indices[family], dtype=np.int64)
            alternatives = members[members != lead]
            pair_seed = paired_support_seed(campaign_id, base_seed=seed)
            rng = np.random.default_rng(pair_seed)
            reserve = np.r_[lead, rng.choice(alternatives, size=max(SUPPORT_SIZES)-1,
                                             replace=False)]
            labels = observations_by_split[split][campaign_index]
            feasible = np.flatnonzero((labels[:, 0] >= GOAL[0]) &
                                      (labels[:, 1] >= GOAL[1]))
            feasible = np.intersect1d(feasible, members)
            unmeasured = np.setdiff1d(feasible, reserve)
            feasibility.append({"split": split, "campaign_id": campaign_id,
                                "family": family, "lead_index": lead,
                                "library_size": len(members),
                                "feasible_library": len(feasible),
                                "feasible_outside_max_support": len(unmeasured),
                                "support_seed": pair_seed,
                                "reserve_indices": reserve.tolist()})
    feasibility_path = output_dir / "feasibility.json"
    feasibility_path.write_text(json.dumps({"goal": GOAL.tolist(), "cases": feasibility},
                                           indent=2, sort_keys=True) + "\n")
    manifest["feasibility_file"] = feasibility_path.name
    manifest["feasibility_sha256"] = _digest(feasibility_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class SteeringDataset:
    """Public fixed-goal episodes with family-local, withheld queries."""

    def __init__(self, data_dir, split: str = "train"):
        if split == "val":
            split = "validation"
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        self.data_dir, self.split = Path(data_dir), split
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        path = self.data_dir / self.manifest["splits"][split]["file"]
        if _digest(path) != self.manifest["splits"][split]["sha256"]:
            raise ValueError("steering split integrity check failed")
        self.graph_dataset = GraphDataset(path)
        with np.load(path, allow_pickle=False) as arrays:
            self.atoms, self.bonds = arrays["atoms"].copy(), arrays["bonds"].copy()
            self.counts = arrays["counts"].copy()
            self.smiles, self.ids = tuple(arrays["smiles"].tolist()), tuple(arrays["ids"].tolist())
            self.observations = arrays["observations"].copy()
            self.campaign_ids = tuple(arrays["campaign_ids"].tolist())
            self.context = arrays["context"].copy()
        self.support_sizes = tuple(self.manifest["support_sizes"])
        self.max_nodes = self.manifest["max_nodes"]
        self.count_prior = np.asarray(self.manifest["count_prior"], dtype=np.float64)
        self.family_leads = tuple(self.manifest["family_leads"])
        self.family_indices = tuple(np.asarray(indices, dtype=np.int64)
                                    for indices in self.manifest["family_indices"])
        self.eligible_leads = np.asarray(self.family_leads, dtype=np.int64)
        meta = self.manifest["splits"][split]
        self.campaign_pairs = meta["campaign_pairs"]
        self.campaign_family = meta["campaign_family"]
        self.campaign_lead_indices = {campaign: self.family_leads[family]
                                      for campaign, family in self.campaign_family.items()}

    def __len__(self):
        return len(self.counts)

    def graph(self, index: int) -> dict:
        return {"atoms": self.atoms[index], "bonds": self.bonds[index],
                "count": int(self.counts[index]), "smiles": self.smiles[index],
                "id": self.ids[index]}

    def paired_support_seed(self, campaign_id: str) -> int:
        if campaign_id not in self.campaign_pairs:
            raise KeyError(campaign_id)
        return paired_support_seed(campaign_id, base_seed=self.manifest["seed"])

    def sample_episode(self, rng: np.random.Generator, *, support_size: int | None = None,
                       queries_per_lead: int = 4, campaign_index: int | None = None,
                       lead_index: int | None = None) -> EpisodeRecords:
        size = int(rng.choice(self.support_sizes)) if support_size is None else int(support_size)
        if size not in self.support_sizes or queries_per_lead < 1:
            raise ValueError("invalid support/query size")
        campaign_index = (int(rng.integers(len(self.campaign_ids))) if campaign_index is None
                          else int(campaign_index))
        campaign_id = self.campaign_ids[campaign_index]
        family = self.campaign_family[campaign_id]
        lead = self.family_leads[family] if lead_index is None else int(lead_index)
        if lead != self.family_leads[family]:
            raise ValueError("campaign requires its declared family lead")
        others = self.family_indices[family]
        others = others[others != lead]
        if size - 1 + queries_per_lead > len(others):
            raise ValueError("family too small for disjoint support/query")
        selected = rng.choice(others, size=size - 1 + queries_per_lead, replace=False)
        support = np.r_[lead, selected[:size - 1]].astype(np.int64)
        queries = selected[size - 1:].astype(np.int64)
        y0 = self.observations[campaign_index, lead].copy()
        query_values = self.observations[campaign_index, queries].copy()
        delta = query_values - y0
        goals = np.repeat(GOAL[None, :], len(queries), axis=0)
        weights = ((delta[:, 0] >= GOAL[0]) & (delta[:, 1] >= GOAL[1])).astype(np.float32)
        return EpisodeRecords(
            [self.graph(index) for index in support],
            self.observations[campaign_index, support].copy(), self.graph(lead), y0,
            [self.graph(index) for index in queries], query_values, delta, goals, weights,
            self.context.copy(), campaign_id, support, lead, queries, self.split)

    def nested_support(self, campaign_index: int, lead_index: int, *, seed: int,
                       sizes=SUPPORT_SIZES) -> dict:
        sizes = tuple(sorted(set(map(int, sizes))))
        if not sizes or min(sizes) < 1 or max(sizes) > max(self.support_sizes):
            raise ValueError("unsupported nested support sizes")
        campaign_id = self.campaign_ids[int(campaign_index)]
        family = self.campaign_family[campaign_id]
        if int(lead_index) != self.family_leads[family]:
            raise ValueError("campaign requires its declared family lead")
        others = self.family_indices[family]
        others = others[others != lead_index]
        rng = np.random.default_rng(seed)
        reserve = np.r_[lead_index, rng.choice(others, size=max(sizes)-1,
                                               replace=False)].astype(np.int64)
        records = {size: {"graphs": [self.graph(index) for index in reserve[:size]],
                          "values": self.observations[campaign_index, reserve[:size]].copy(),
                          "context": self.context.copy(), "indices": reserve[:size].copy()}
                   for size in sizes}
        return {"records": records, "reserve_indices": reserve,
                "reserve_smiles": tuple(self.smiles[index] for index in reserve),
                "reserve_ids": tuple(self.ids[index] for index in reserve),
                "campaign_id": campaign_id, "lead": self.graph(lead_index),
                "lead_values": self.observations[campaign_index, lead_index].copy()}

    def sample_batch(self, rng: np.random.Generator, *, episodes: int = 8,
                     queries_per_lead: int = 4):
        return [self.sample_episode(rng, queries_per_lead=queries_per_lead)
                for _ in range(episodes)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="2_data_storage/processed/evidence_steering_toy")
    parser.add_argument("--seed", type=int, default=142)
    args = parser.parse_args()
    manifest = prepare_steering_data(args.output_dir, seed=args.seed)
    print(json.dumps({"families": manifest["family_sizes"],
                      "molecules": manifest["splits"]["train"]["size"],
                      "campaigns": {split: len(manifest["splits"][split]["campaign_ids"])
                                    for split in SPLITS}}, indent=2))


if __name__ == "__main__":
    main()
