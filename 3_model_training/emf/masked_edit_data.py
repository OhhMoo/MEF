"""Lead-declared masked edits and molecule-held-out response campaigns.

The editable coordinates are fixed from the two lead structures below. Query
identity and response values are never inputs to mask construction. Existing
exact alignments are chemistry-only and are reused without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, rdMolDescriptors

from .local_edit_data import DEFAULT_ALIGNMENT_PATH
from .steering_data import SteeringDataset


SOURCE_DIR = Path("2_data_storage/processed/evidence_steering_toy")
DEFAULT_DIR = Path("2_data_storage/processed/masked_edit_toy")
SPLITS = ("train", "validation", "test")
MAX_NODES = 9
UPPER = np.triu_indices(MAX_NODES, k=1)
# Coordinates refer to each canonical lead, whose graph and identity are pinned
# by the source manifest. These masks represent terminal branches, with the
# central or boundary bond to the protected scaffold held fixed.
LEAD_MASK_ATOMS = {
    0: {"left": (0, 1, 2, 3, 4), "right": (5, 6, 7, 8)},
    1: {"left": (0, 1, 2, 3), "right": (5, 6, 7, 8)},
}
EXPECTED_LEADS = {0: "CCNC(=O)CC(C)C", 1: "CCC(=O)CNC(C)=O"}
SCALAR_GOAL = (0.25, -0.10)
VECTOR_GOAL = (0.25, -0.10)
SUPPORT_SIZES = (4, 8, 16)
DESCRIPTORS = ("RDKit MolLogP", "RDKit TPSA")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid molecule for descriptor")
    return np.asarray((Crippen.MolLogP(molecule),
                       rdMolDescriptors.CalcTPSA(molecule)), dtype=np.float64)


def _features(raw: np.ndarray, lead_raw: np.ndarray, scales: np.ndarray) -> np.ndarray:
    scaled = raw / scales
    lead_scaled = lead_raw / scales
    phi = np.stack((scaled[..., 0], scaled[..., 1],
                    scaled[..., 0] * scaled[..., 1]), axis=-1)
    lead_phi = np.asarray((lead_scaled[0], lead_scaled[1],
                           lead_scaled[0] * lead_scaled[1]))
    return phi - lead_phi


def _response(raw: np.ndarray, lead_raw: np.ndarray, scales: np.ndarray,
              coefficients: np.ndarray) -> np.ndarray:
    """Exact, lead-centered two-endpoint function; no table lookup."""
    return _features(raw, lead_raw, scales) @ coefficients.T


def _campaign_coefficients(split: str, family: int, mask_name: str,
                           kind: str, number: int, seed: int, attempt: int = 0) -> np.ndarray:
    key = f"{seed}:{split}:{family}:{mask_name}:{kind}:{number}:{attempt}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    if kind == "scalar":
        # Historical two-endpoint relation for the controlled first comparison.
        magnitude = rng.uniform(0.55, 1.85)
        # Source-only feasibility audit found no positive-sign candidate in
        # family 0's right-branch mask at the declared goal. The scalar sanity
        # arm therefore uses its feasible negative orientation there.
        sign = -1.0 if (family == 0 and mask_name == "right") else (-1.0, 1.0)[number % 2]
        return np.asarray(((sign * magnitude, 0, 0),
                           (0, -1, 0)), dtype=np.float64)
    if kind == "vector":
        # Independent endpoint variation plus a lead-centered nonlinear term.
        # Terms are seeded independently for every source/held-out campaign.
        linear = rng.uniform(-1.4, 1.4, (2, 2))
        nonlinear = rng.uniform(-0.20, 0.20, (2, 1))
        return np.concatenate((linear, nonlinear), axis=1)
    raise ValueError(f"unknown campaign kind: {kind}")


@dataclass(frozen=True)
class EditMask:
    mask_id: str
    family: int
    name: str
    atom_editable: np.ndarray
    bond_editable: np.ndarray
    lead_atoms: np.ndarray
    lead_bonds: np.ndarray
    lead_smiles: str
    lead_index: int

    @property
    def nodes(self) -> np.ndarray:
        return self.atom_editable

    @property
    def pairs(self) -> np.ndarray:
        return self.bond_editable

    def as_dict(self) -> dict:
        return {"mask_id": self.mask_id, "family": self.family, "name": self.name,
                "atom_editable": self.atom_editable.copy(),
                "bond_editable": self.bond_editable.copy(),
                "lead_atoms": self.lead_atoms.copy(), "lead_bonds": self.lead_bonds.copy(),
                "lead_smiles": self.lead_smiles, "lead_index": self.lead_index}


@dataclass(frozen=True)
class MaskedEpisode:
    support_graphs: list[dict]
    support_values: np.ndarray
    lead_graph: dict
    lead_values: np.ndarray
    query_graphs: list[dict]
    query_values: np.ndarray
    deltas: np.ndarray
    goals: np.ndarray
    flow_weights: np.ndarray
    context: np.ndarray
    campaign_id: str
    support_indices: np.ndarray
    lead_index: int
    query_indices: np.ndarray
    split: str
    mask_id: str
    atom_editable: np.ndarray
    bond_editable: np.ndarray


def _make_masks(source: SteeringDataset) -> dict[str, EditMask]:
    masks = {}
    for family, declarations in LEAD_MASK_ATOMS.items():
        lead_index = int(source.family_leads[family])
        if source.smiles[lead_index] != EXPECTED_LEADS[family]:
            raise ValueError(f"lead identity changed for family {family}")
        lead_atoms = source.atoms[lead_index].copy()
        lead_bonds = source.bonds[lead_index].copy()
        for name, atom_positions in declarations.items():
            atom_editable = np.zeros(MAX_NODES, dtype=bool)
            atom_editable[list(atom_positions)] = True
            bond_editable = atom_editable[UPPER[0]] & atom_editable[UPPER[1]]
            boundary = ((atom_editable[UPPER[0]] ^ atom_editable[UPPER[1]]) &
                        (lead_bonds != 0))
            if np.count_nonzero(boundary) != 1:
                raise ValueError(f"declared mask is not a single attached branch: {family}/{name}")
            mask_id = f"f{family}_{name}"
            masks[mask_id] = EditMask(mask_id, family, name, atom_editable,
                                      bond_editable, lead_atoms.copy(),
                                      lead_bonds.copy(), source.smiles[lead_index], lead_index)
    return masks


def _compatible(graph: dict, mask: EditMask) -> bool:
    return (int(graph["count"]) == MAX_NODES and
            not np.any((np.asarray(graph["atoms"]) != mask.lead_atoms) & ~mask.atom_editable) and
            not np.any((np.asarray(graph["bonds"]) != mask.lead_bonds) & ~mask.bond_editable))


def _molecule_splits(mask_members: dict[str, list[int]], seed: int,
                     total_molecules: int) -> dict[str, list[int]]:
    """One global identity split, stratified using chemistry-only mask membership."""
    rng = np.random.default_rng(seed)
    allocation: dict[int, str] = {}
    for mask_id in sorted(mask_members, key=lambda key: (len(mask_members[key]), key)):
        members = np.asarray(mask_members[mask_id], dtype=np.int64)
        # Reserve at least five analogues for each held-out split. A support of
        # lead plus three analogues then leaves at least two query molecules.
        target = max(5, int(np.ceil(0.20 * len(members))))
        if len(members) - 2 * target < 8:
            raise ValueError(f"insufficient masked analogues: {mask_id}")
        for split in ("validation", "test"):
            current = sum(allocation.get(int(index)) == split for index in members)
            available = [int(index) for index in members if int(index) not in allocation]
            need = max(0, target - current)
            if need > len(available):
                raise ValueError(f"cannot form global molecule split for {mask_id}")
            for index in rng.choice(available, size=need, replace=False):
                allocation[int(index)] = split
    for index in range(total_molecules):
        allocation.setdefault(index, "train")
    return {split: sorted(index for index, assigned in allocation.items() if assigned == split)
            for split in SPLITS}


def prepare_masked_edit_data(output_dir=DEFAULT_DIR, *, source_dir=SOURCE_DIR,
                             alignment_path=DEFAULT_ALIGNMENT_PATH, seed: int = 142,
                             observation_sigma: float = 0.05) -> dict:
    """Build fixed chemistry, molecule splits and independent campaign observations."""
    output_dir, source_dir, alignment_path = map(Path, (output_dir, source_dir, alignment_path))
    if observation_sigma < 0:
        raise ValueError("observation_sigma must be nonnegative")
    source = SteeringDataset(source_dir, "train")
    alignment = json.loads(alignment_path.read_text())
    if (alignment.get("schema_version") != 1 or alignment.get("max_nodes") != MAX_NODES or
            alignment.get("family_leads") != list(source.family_leads)):
        raise ValueError("alignment/source mismatch")
    masks = _make_masks(source)
    aligned = {mask_id: {
        int(index): alignment["families"][str(mask.family)][str(index)]
        for index in source.family_indices[mask.family]}
        for mask_id, mask in masks.items()}
    mask_members = {mask_id: [index for index, graph in records.items()
                              if index != masks[mask_id].lead_index and
                              _compatible(graph, masks[mask_id])]
                    for mask_id, records in aligned.items()}
    molecule_split = _molecule_splits(mask_members, seed, len(source.smiles))
    raw = np.stack([_descriptor(smiles) for smiles in source.smiles])
    scales = np.std(raw[np.asarray(molecule_split["train"], dtype=int)], axis=0).clip(1e-6)
    campaigns = {}
    observations = {}
    coefficients = {}
    feasibility = []
    family_members = {family: set(map(int, source.family_indices[family]))
                      for family in LEAD_MASK_ATOMS}
    source_set = set(molecule_split["train"])
    for split in SPLITS:
        for mask_id, mask in masks.items():
            split_set = set(molecule_split[split])
            indices = [index for index in mask_members[mask_id] if index in split_set]
            support_pool = sorted((split_set & family_members[mask.family]) - {mask.lead_index})
            if len(indices) < (8 if split == "train" else 5):
                raise ValueError(f"insufficient {split} molecules for {mask_id}: {len(indices)}")
            if len(support_pool) < max(SUPPORT_SIZES)-1:
                raise ValueError(f"insufficient {split} family support pool for {mask_id}")
            for kind, count in (("scalar", 8 if split == "train" else 2),
                                ("vector", 16 if split == "train" else 4)):
                for number in range(count):
                    cid = f"{split}_{mask_id}_{kind}_{number:02d}"
                    goal = SCALAR_GOAL if kind == "scalar" else VECTOR_GOAL
                    source_members = np.asarray(sorted(source_set & set(mask_members[mask_id])),
                                                dtype=np.int64)
                    desired_floor = 1 if mask_id == "f0_left" and kind == "scalar" else 2
                    best = None
                    for attempt in range(2000):
                        coeff = _campaign_coefficients(split, mask.family, mask.name,
                                                       kind, number, seed, attempt)
                        source_exact = _response(raw[source_members], raw[mask.lead_index],
                                                 scales, coeff)
                        source_feasible = int(np.count_nonzero(
                            np.all(source_exact >= np.asarray(goal), axis=1)))
                        if best is None or source_feasible > best[0]:
                            best = (source_feasible, coeff.copy(), attempt)
                        if source_feasible >= desired_floor:
                            break
                    else:
                        if best is None or best[0] < 1:
                            raise ValueError(f"could not draw source-feasible context for {cid}")
                        source_feasible, coeff, attempt = best
                    exact = _response(raw, raw[mask.lead_index], scales, coeff)
                    noise_seed = int.from_bytes(hashlib.sha256(f"noise:{seed}:{cid}".encode()).digest()[:8], "little")
                    observed = exact + np.random.default_rng(noise_seed).normal(
                        0.0, observation_sigma, exact.shape)
                    campaigns[cid] = {"split": split, "mask_id": mask_id, "family": mask.family,
                                      "kind": kind, "goal": list(goal),
                                      "lead_smiles": mask.lead_smiles,
                                      "source_feasible_count": source_feasible,
                                      "source_feasibility_floor": desired_floor if source_feasible >= desired_floor else 1,
                                      "source_context_draw_attempt": attempt,
                                      "molecule_indices": sorted(indices),
                                      "support_pool_indices": support_pool}
                    # Preserve the full chemistry axis for stable graph indexing,
                    # while withholding all response labels outside this molecule
                    # split. The common lead is a declared zero-reference exception.
                    visible = np.zeros(len(source.smiles), dtype=bool)
                    visible[support_pool] = True
                    visible[mask.lead_index] = True
                    observed[~visible] = np.nan
                    observations[cid] = observed.astype(np.float32)
                    coefficients[cid] = coeff.tolist()
                    feasible = [index for index in indices
                                if np.all(exact[index] - exact[mask.lead_index] >= goal)]
                    feasibility.append({"campaign_id": cid, "compatible_molecules": len(indices),
                                        "feasible_molecules": len(feasible)})
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(exist_ok=True)
    oracle_path = private_dir / "oracle.json"
    oracle_path.write_text(json.dumps({"schema_version": 1, "coefficients": coefficients,
                                       "scales": scales.tolist(), "descriptors": DESCRIPTORS,
                                       "campaigns": campaigns, "all_split_feasibility": feasibility},
                                      indent=2, sort_keys=True) + "\n")
    chemistry_path = output_dir / "chemistry.npz"
    np.savez_compressed(chemistry_path, atoms=source.atoms, bonds=source.bonds,
                        counts=source.counts, smiles=np.asarray(source.smiles),
                        ids=np.asarray(source.ids))
    for split in SPLITS:
        ids = [cid for cid, meta in campaigns.items() if meta["split"] == split]
        np.savez_compressed(output_dir / f"{split}.npz", campaign_ids=np.asarray(ids),
                            observations=np.stack([observations[cid] for cid in ids]))
    manifest = {"schema_version": 1, "dataset": "masked_edit_toy",
                "seed": seed, "max_nodes": MAX_NODES, "rdkit_version": rdBase.rdkitVersion,
                "observation_sigma": observation_sigma,
                "source_manifest_sha256": _sha256(source_dir / "manifest.json"),
                "alignment_sha256": _sha256(alignment_path),
                "chemistry_sha256": _sha256(chemistry_path),
                "oracle_sha256": _sha256(oracle_path),
                "splits_sha256": {split: _sha256(output_dir / f"{split}.npz") for split in SPLITS},
                "molecule_splits": molecule_split, "support_sizes": SUPPORT_SIZES,
                "descriptor_scales_from_source_molecules": scales.tolist(),
                "response_features": "phi=[MolLogP/scale0,TPSA/scale1,(MolLogP/scale0)*(TPSA/scale1)]; exact delta=W@(phi(G)-phi(lead))",
                "goals": {"scalar": SCALAR_GOAL, "vector": VECTOR_GOAL},
                "campaigns": campaigns,
                "masks": {mask_id: {"family": mask.family, "name": mask.name,
                                     "lead_index": mask.lead_index,
                                     "lead_smiles": mask.lead_smiles,
                                     "atom_editable": mask.atom_editable.tolist(),
                                     "bond_editable": mask.bond_editable.tolist(),
                                     "compatible_count": len(mask_members[mask_id])}
                          for mask_id, mask in masks.items()},
                "source_feasibility": [record for record in feasibility
                                       if record["campaign_id"].startswith("train_")]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class MaskedEditDataset:
    """Episodes with molecule-held-out compatible queries and disjoint supports."""

    def __init__(self, data_dir=DEFAULT_DIR, split: str = "train", *, kind: str | None = None,
                 verify: bool = True, alignment_path=DEFAULT_ALIGNMENT_PATH):
        if split == "val":
            split = "validation"
        if split not in SPLITS:
            raise ValueError(split)
        self.data_dir, self.split = Path(data_dir), split
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        if verify:
            for filename, digest in (("chemistry.npz", self.manifest["chemistry_sha256"]),
                                     (f"{split}.npz", self.manifest["splits_sha256"][split])):
                if _sha256(self.data_dir / filename) != digest:
                    raise ValueError(f"data integrity check failed: {filename}")
        with np.load(self.data_dir / "chemistry.npz", allow_pickle=False) as records:
            self.atoms, self.bonds, self.counts = (records[key].copy() for key in
                                                  ("atoms", "bonds", "counts"))
            self.smiles = tuple(records["smiles"].tolist())
            self.ids = tuple(records["ids"].tolist())
        with np.load(self.data_dir / f"{split}.npz", allow_pickle=False) as records:
            ids = records["campaign_ids"].tolist()
            values = records["observations"]
        self.campaign_ids = tuple(cid for cid in ids
                                  if kind is None or self.manifest["campaigns"][cid]["kind"] == kind)
        self.observations = {cid: values[ids.index(cid)].copy() for cid in self.campaign_ids}
        self.masks = {}
        for mask_id, record in self.manifest["masks"].items():
            lead = record["lead_index"]
            self.masks[mask_id] = EditMask(mask_id, record["family"], record["name"],
                                           np.asarray(record["atom_editable"], bool),
                                           np.asarray(record["bond_editable"], bool),
                                           self.atoms[lead].copy(), self.bonds[lead].copy(),
                                           record["lead_smiles"], lead)
        alignment_path = Path(alignment_path)
        if verify and _sha256(alignment_path) != self.manifest["alignment_sha256"]:
            raise ValueError("alignment integrity check failed")
        self.alignment = json.loads(alignment_path.read_text())
        self.max_nodes = MAX_NODES
        self.support_sizes = tuple(self.manifest["support_sizes"])
        self.context = np.eye(2, dtype=np.float32).reshape(-1)
        self.molecule_splits = {name: set(map(int, indices)) for name, indices in
                                self.manifest["molecule_splits"].items()}
        self.graph_indices = np.asarray(sorted(self.molecule_splits[split]), dtype=np.int64)
        self.source_graph_indices = np.asarray(sorted(self.molecule_splits["train"]), dtype=np.int64)
        self.tasks = tuple({"task_id": mask_id, "family": mask.family,
                            "lead_index": mask.lead_index,
                            "atom_editable": mask.atom_editable.copy(),
                            "bond_editable": mask.bond_editable.copy(),
                            "compatible_indices": {name: np.asarray(sorted(
                                set(map(int, self.manifest["molecule_splits"][name])) &
                                {int(index) for cid, meta in self.manifest["campaigns"].items()
                                 if meta["mask_id"] == mask_id and meta["split"] == name
                                 for index in meta["molecule_indices"]}), dtype=np.int64)
                                for name in SPLITS}}
                           for mask_id, mask in self.masks.items())

    def graph(self, index: int, *, aligned: bool = False,
              task_id: str | None = None, mask_id: str | None = None) -> dict:
        graph = {"atoms": self.atoms[index].copy(), "bonds": self.bonds[index].copy(),
                 "count": int(self.counts[index]), "smiles": self.smiles[index],
                 "id": self.ids[index]}
        if task_id is not None:
            mask_id = task_id
        if aligned or mask_id is not None:
            if mask_id is None:
                raise ValueError("aligned graph requires task_id")
            mask = self.masks[mask_id]
            record = self.alignment["families"][str(mask.family)][str(index)]
            graph.update(atoms=np.asarray(record["atoms"], np.uint8),
                         bonds=np.asarray(record["bonds"], np.uint8),
                         mask_id=mask_id)
        return graph

    def sample_episode(self, rng: np.random.Generator, *, support_size: int = 4,
                       queries_per_lead: int = 2, campaign_index: int | None = None,
                       lead_index: int | None = None) -> MaskedEpisode:
        cid = self.campaign_ids[int(rng.integers(len(self.campaign_ids))) if campaign_index is None
                                else int(campaign_index)]
        meta = self.manifest["campaigns"][cid]
        mask = self.masks[meta["mask_id"]]
        if lead_index is not None and lead_index != mask.lead_index:
            raise ValueError("wrong lead for campaign")
        if support_size not in self.support_sizes or queries_per_lead < 1:
            raise ValueError("unsupported episode size")
        members = np.asarray(meta["molecule_indices"], dtype=np.int64)
        pool = np.asarray(meta["support_pool_indices"], dtype=np.int64)
        if len(members) < queries_per_lead or len(pool) < support_size - 1 + queries_per_lead:
            raise ValueError("too few compatible molecules")
        queries = rng.choice(members, size=queries_per_lead, replace=False).astype(np.int64)
        support_pool = np.setdiff1d(pool, queries)
        support = np.r_[mask.lead_index, rng.choice(support_pool, size=support_size - 1,
                                                    replace=False)].astype(np.int64)
        values = self.observations[cid]
        y0 = values[mask.lead_index].copy()
        query_values = values[queries].copy()
        delta = query_values - y0
        goals = np.repeat(np.asarray(meta["goal"], np.float32)[None, :], len(queries), axis=0)
        weights = np.all(delta >= goals, axis=1).astype(np.float32)
        return MaskedEpisode([self.graph(int(i)) for i in support], values[support].copy(),
                             self.graph(mask.lead_index), y0,
                             [self.graph(int(i), mask_id=mask.mask_id) for i in queries],
                             query_values, delta, goals, weights, self.context.copy(), cid,
                             support, mask.lead_index, queries, self.split, mask.mask_id,
                             mask.atom_editable.copy(), mask.bond_editable.copy())

    def nested_support(self, campaign_index: int, *, seed: int, sizes=SUPPORT_SIZES) -> dict:
        cid = self.campaign_ids[int(campaign_index)]
        meta = self.manifest["campaigns"][cid]
        mask = self.masks[meta["mask_id"]]
        sizes = tuple(sorted(set(map(int, sizes))))
        if not sizes or min(sizes) < 1 or max(sizes) > max(self.support_sizes):
            raise ValueError("unsupported support sizes")
        members = np.asarray(meta["molecule_indices"], dtype=np.int64)
        pool = np.asarray(meta["support_pool_indices"], dtype=np.int64)
        rng = np.random.default_rng(seed)
        outside = rng.permutation(np.setdiff1d(pool, members))
        inside = rng.permutation(members)
        reserve = np.r_[mask.lead_index,
                        np.r_[outside, inside][:max(sizes)-1]].astype(np.int64)
        values = self.observations[cid]
        return {"records": {size: {"graphs": [self.graph(int(i)) for i in reserve[:size]],
                                     "values": values[reserve[:size]].copy(),
                                     "indices": reserve[:size].copy(),
                                     "context": self.context.copy()}
                            for size in sizes},
                "reserve_indices": reserve, "reserve_smiles": tuple(self.smiles[i] for i in reserve),
                "campaign_id": cid, "lead": self.graph(mask.lead_index),
                "lead_values": values[mask.lead_index].copy(), "mask": mask.as_dict(),
                "query_indices": np.asarray(sorted(set(members) - set(reserve)), dtype=np.int64)}

    def compatible_library(self, campaign_id: str, *, exclude_indices=()) -> list[dict]:
        """Finite compatible-library control, independent of response outcomes."""
        meta = self.manifest["campaigns"][campaign_id]
        excluded = set(map(int, exclude_indices))
        return [self.graph(index, mask_id=meta["mask_id"])
                for index in meta["molecule_indices"] if index not in excluded]

    def sample_pairs(self, rng: np.random.Generator, batch_size: int,
                     *, split: str | None = None) -> list[dict]:
        """Draw trainable aligned lead/query pairs from a declared split."""
        if split is not None and split != self.split:
            raise ValueError("load the requested split before sampling")
        tasks = [task for task in self.tasks if len(task["compatible_indices"][self.split])]
        pairs = []
        for _ in range(batch_size):
            task = tasks[int(rng.integers(len(tasks)))]
            index = int(rng.choice(task["compatible_indices"][self.split]))
            pairs.append({"task_id": task["task_id"], "family": task["family"],
                          "lead_index": task["lead_index"], "query_index": index,
                          "lead_graph": self.graph(task["lead_index"]),
                          "query_graph": self.graph(index, aligned=True,
                                                    task_id=task["task_id"]),
                          "atom_editable": task["atom_editable"].copy(),
                          "bond_editable": task["bond_editable"].copy()})
        return pairs


def enumerate_constrained_mutations(mask: EditMask, *, max_candidates: int | None = None):
    """Simple one-category mutation control, including invalid attempts."""
    from .chem import decode_graph
    emitted = 0
    for category, positions, base, count in (("atom", np.flatnonzero(mask.atom_editable), mask.lead_atoms, 4),
                                             ("bond", np.flatnonzero(mask.bond_editable), mask.lead_bonds, 4)):
        for position in positions:
            for label in range(count):
                if label == int(base[position]):
                    continue
                atoms, bonds = mask.lead_atoms.copy(), mask.lead_bonds.copy()
                (atoms if category == "atom" else bonds)[position] = label
                outcome = decode_graph(atoms, bonds, MAX_NODES, MAX_NODES)
                yield {"atoms": atoms, "bonds": bonds, "count": MAX_NODES,
                       "smiles": outcome.smiles, "chemically_valid": outcome.chemically_valid,
                       "connected": outcome.connected, "failure_reason": outcome.failure_reason,
                       "mutation_type": category, "position": int(position), "label": label}
                emitted += 1
                if max_candidates is not None and emitted >= max_candidates:
                    return


def fit_support_response(support_smiles, support_values, lead_smiles: str,
                         scales, *, ridge: float = 0.05) -> dict:
    """Estimate the campaign map from observed support only.

    This is a transparent finite-library/control diagnostic. It uses the
    declared descriptor basis, never the hidden coefficient matrix or a query
    outcome. Rank and singular values expose weak support geometry.
    """
    smiles = tuple(support_smiles)
    values = np.asarray(support_values, dtype=np.float64)
    if values.shape != (len(smiles), 2) or not np.all(np.isfinite(values)):
        raise ValueError("support_values must be finite [N,2]")
    if lead_smiles not in smiles:
        raise ValueError("support must include the declared lead")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    lead_row = smiles.index(lead_smiles)
    design = _features(np.stack([_descriptor(item) for item in smiles]),
                       _descriptor(lead_smiles), np.asarray(scales, np.float64))
    response = values - values[lead_row]
    singular = np.linalg.svd(design, compute_uv=False)
    coefficients = np.linalg.solve(design.T @ design + max(ridge, 1e-12) * np.eye(3),
                                   design.T @ response).T
    return {"coefficients": coefficients, "basis_rank": int(np.linalg.matrix_rank(design)),
            "singular_values": singular,
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
            "support_count": len(smiles), "ridge": ridge}
