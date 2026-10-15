"""Reproducible support/query episodes for experimentally calibrated redesign.

The public dataset contains observations and graph records. The coefficient
file is loaded only by preparation or an external evaluator, never by episode
sampling. Compound splits precede neighbourhood pairs and campaign episodes.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .data import GraphDataset, collate_graphs
from .redesign_contracts import CampaignBatch, SupportBatch
from .toy_oracle import FEATURE_NAMES, ToyOracle, structural_features


SPLITS = ("train", "validation", "test")
DEFAULT_SUPPORT_SIZES = (16, 32, 64, 128)
CONTEXT = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def file_digest(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecular_fingerprint(smiles: str):
    # structural_features is the supported-chemistry validator in public helpers.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_smiles")
    return _FINGERPRINT_GENERATOR.GetFingerprint(mol)


def neighbourhood(smiles: str, lead_smiles: str, *, min_similarity: float = 0.4,
                  max_count_difference: int = 2) -> dict:
    """Declared local redesign rule, independent of endpoint predictions."""
    candidate_features = structural_features(smiles)
    lead_features = structural_features(lead_smiles)
    count_difference = abs(round(candidate_features[0] * 9) -
                           round(lead_features[0] * 9))
    similarity = float(DataStructs.TanimotoSimilarity(
        molecular_fingerprint(smiles), molecular_fingerprint(lead_smiles)))
    return {"similarity": similarity, "count_difference": count_difference,
            "eligible": similarity >= min_similarity and
                        count_difference <= max_count_difference}


def neighbour_pairs(fingerprints, counts: np.ndarray, *, min_similarity: float = 0.4,
                    max_count_difference: int = 2, max_neighbours: int = 4):
    """Nearest compatible analogues; identity and stable-index break ties."""
    if not 0 <= min_similarity <= 1 or min(max_count_difference, max_neighbours) < 0:
        raise ValueError("invalid neighbour settings")
    n = len(counts)
    neighbours = np.full((n, max_neighbours), -1, dtype=np.int32)
    similarities = np.zeros_like(neighbours, dtype=np.float32)
    indices = np.arange(n)
    counts = counts.astype(np.int16)
    for index, fingerprint in enumerate(fingerprints):
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints))
        eligible = np.flatnonzero((values >= min_similarity) & (indices != index) &
                                 (np.abs(counts - counts[index]) <= max_count_difference))
        order = np.lexsort((eligible, -values[eligible]))[:max_neighbours]
        selected = eligible[order]
        neighbours[index, :len(selected)] = selected
        similarities[index, :len(selected)] = values[selected]
    return neighbours, similarities


def conditional_count_prior(counts: np.ndarray, neighbours: np.ndarray,
                            max_nodes: int = 9) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Fit source-pair count rows before any goal or response filtering."""
    histogram = np.zeros((max_nodes, max_nodes), dtype=np.int64)
    for lead, candidates in enumerate(neighbours):
        for candidate in candidates[candidates >= 0]:
            histogram[int(counts[lead]) - 1, int(counts[candidate]) - 1] += 1
    prior = histogram.astype(np.float64)
    unsupported = np.flatnonzero(prior.sum(axis=1) == 0)
    # An absent source row is unsupported, not an invented fitted distribution.
    prior /= prior.sum(axis=1, keepdims=True).clip(1)
    return prior, histogram, (unsupported + 1).tolist()


def draw_goals(rng: np.random.Generator, lead_values: np.ndarray, number: int, *,
               s1: float = 1.0, s2: float = 1.0) -> np.ndarray:
    """Outcome-independent request prior; no query argument can enter it."""
    if s1 <= 0 or s2 <= 0 or number < 1:
        raise ValueError("positive goal scales and query count required")
    kappa = rng.uniform(-s1, s1, size=number)
    tau = rng.uniform(float(lead_values[1]) - s2, float(lead_values[1]), size=number)
    return np.column_stack((kappa, tau)).astype(np.float32)


def feasibility_weights(delta: np.ndarray, lead_values: np.ndarray,
                        goals: np.ndarray) -> np.ndarray:
    return ((delta[:, 0] >= goals[:, 0]) &
            (float(lead_values[1]) + delta[:, 1] >= goals[:, 1])).astype(np.float32)


def prepare_campaigns(graph_dir, output_dir, *, seed: int = 42,
                      source_campaigns: int = 32, validation_campaigns: int = 8,
                      test_campaigns: int = 8, support_sizes=DEFAULT_SUPPORT_SIZES,
                      min_similarity: float = 0.4, max_count_difference: int = 2,
                      max_neighbours: int = 4, molecule_limit: int | None = None) -> dict:
    """Cache graphs/features/pairs and exact complete-case campaign observations."""
    graph_dir, output_dir = Path(graph_dir), Path(output_dir)
    support_sizes = tuple(sorted(set(int(size) for size in support_sizes)))
    if not support_sizes or min(support_sizes) < 1:
        raise ValueError("support sizes must be positive")
    config = {"seed": seed, "source_campaigns": source_campaigns,
              "validation_campaigns": validation_campaigns, "test_campaigns": test_campaigns,
              "support_sizes": list(support_sizes), "min_similarity": min_similarity,
              "max_count_difference": max_count_difference, "max_neighbours": max_neighbours,
              "molecule_limit": molecule_limit, "unsupported_count_policy": "zero-row"}
    inputs = {split: file_digest(graph_dir / f"{split}.npz") for split in SPLITS}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["config"] != config or existing["input_sha256"] != inputs:
            raise FileExistsError("Campaign output belongs to another configuration; use a new path")
        if all(file_digest(output_dir / item["file"]) == item["sha256"]
               for item in existing["splits"].values()):
            return existing
        raise ValueError("campaign cache integrity check failed")
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    caches, identity_sets = {}, {}
    for split in SPLITS:
        with np.load(graph_dir / f"{split}.npz", allow_pickle=False) as arrays:
            cache = {key: arrays[key].copy() for key in arrays.files}
        if molecule_limit is not None:
            cache = {key: value[:molecule_limit] for key, value in cache.items()}
        if len(cache["counts"]) <= max(support_sizes) + max_neighbours:
            raise ValueError(f"{split} has too few compounds for disjoint support/query episodes")
        identity_sets[split] = set(cache["smiles"].tolist())
        if len(identity_sets[split]) != len(cache["counts"]):
            raise ValueError(f"duplicate compound identity inside {split}")
        features = np.stack([structural_features(smiles, cache["atoms"].shape[1])
                             for smiles in cache["smiles"]])
        fingerprints = [molecular_fingerprint(smiles) for smiles in cache["smiles"]]
        neighbours, similarities = neighbour_pairs(
            fingerprints, cache["counts"], min_similarity=min_similarity,
            max_count_difference=max_count_difference, max_neighbours=max_neighbours)
        cache.update(features=features, neighbours=neighbours, similarities=similarities)
        caches[split] = cache
        print(json.dumps({"event": "graph_cache", "split": split,
                          "compounds": len(features),
                          "eligible_leads": int((neighbours >= 0).any(axis=1).sum()),
                          "seconds": round(time.monotonic() - started, 3)}), flush=True)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if identity_sets[left] & identity_sets[right]:
            raise ValueError(f"compound identity leakage between {left} and {right}")
    max_nodes = caches["train"]["atoms"].shape[1]
    oracle = ToyOracle.create(seed=seed, source=source_campaigns,
                              validation=validation_campaigns, test=test_campaigns,
                              max_nodes=max_nodes)
    oracle.fit_source_normalization(caches["train"]["features"])
    oracle_path = output_dir / "private" / "oracle.json"
    oracle.save(oracle_path)
    prior, histogram, unsupported = conditional_count_prior(
        caches["train"]["counts"], caches["train"]["neighbours"], max_nodes)
    split_info = {}
    for split, cache in caches.items():
        names = [name for name in oracle.campaigns if name.startswith(f"{split}_")]
        outcomes = np.stack([oracle.score_features(name, cache["features"]) for name in names])
        if not np.isfinite(outcomes).all():
            raise ValueError("oracle supplied nonfinite complete-case outcomes")
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **cache, observations=outcomes.astype(np.float32),
                            campaign_ids=np.asarray(names), context=CONTEXT)
        leads = (cache["neighbours"] >= 0).any(axis=1)
        split_info[split] = {"file": path.name, "sha256": file_digest(path),
                             "compounds": len(cache["counts"]), "campaigns": len(names),
                             "eligible_leads": int(leads.sum()),
                             "eligible_lead_fraction": float(leads.mean()),
                             "lead_analogue_pairs": int((cache["neighbours"] >= 0).sum()),
                             "complete_case_fraction": 1.0}
    manifest = {
        "schema_version": 1, "dataset": "restricted_qm9_nonlinear_response_campaigns",
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "input_graph_dir": str(graph_dir.resolve()), "input_sha256": inputs,
        "rdkit_version": rdBase.rdkitVersion, "max_nodes": max_nodes,
        "feature_names": list(FEATURE_NAMES), "context_dim": len(CONTEXT),
        "endpoint_scale": {"units": "dimensionless synthetic response", "larger_is_better": [True, True],
                           "log_transform": "none; synthetic endpoints are signed",
                           "source_mean": oracle.mean.tolist(), "source_std": oracle.std.tolist(),
                           "normalization_fit": "all source campaigns on source training compounds only",
                           "goal_s1": 1.0, "goal_s2": 1.0},
        "count_prior": prior.tolist(), "count_histogram": histogram.tolist(),
        "unsupported_lead_counts": unsupported, "count_fallback": "none; reject unsupported rows",
        "count_fit": "all retained source pairs before any goal filtering",
        "splits": split_info, "private_oracle_sha256": file_digest(oracle_path),
        "oracle_truth": "exact nonlinear graph-feature simulator; coefficients excluded from model inputs",
        "preparation_seconds": time.monotonic() - started,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"event": "campaigns_ready", "manifest": str(manifest_path),
                      "seconds": round(manifest["preparation_seconds"], 3)}), flush=True)
    return manifest


@dataclass(frozen=True)
class EpisodeRecords:
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

    @property
    def support_ids(self):
        return tuple(graph["id"] for graph in self.support_graphs)

    @property
    def query_ids(self):
        return tuple(graph["id"] for graph in self.query_graphs)


class CampaignDataset:
    """Public observation cache; sampling never reads private evaluator metadata."""

    def __init__(self, data_dir, split: str = "train"):
        if split == "val":
            split = "validation"
        if split not in SPLITS:
            raise ValueError(f"unknown campaign split: {split}")
        self.data_dir, self.split = Path(data_dir), split
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        graph_path = self.data_dir / self.manifest["splits"][split]["file"]
        self.graph_dataset = GraphDataset(graph_path)
        with np.load(graph_path, allow_pickle=False) as arrays:
            self.atoms = arrays["atoms"].copy()
            self.bonds = arrays["bonds"].copy()
            self.counts = arrays["counts"].copy()
            self.smiles, self.ids = tuple(arrays["smiles"].tolist()), tuple(arrays["ids"].tolist())
            self.observations = arrays["observations"].copy()
            self.campaign_ids = tuple(arrays["campaign_ids"].tolist())
            self.neighbours = arrays["neighbours"].copy()
            self.similarities = arrays["similarities"].copy()
            self.context = arrays["context"].copy()
        self.support_sizes = tuple(self.manifest["config"]["support_sizes"])
        self.count_prior = np.asarray(self.manifest["count_prior"], dtype=np.float64)
        supported_counts = self.count_prior.sum(axis=1) > 0
        self.eligible_leads = np.flatnonzero((self.neighbours >= 0).any(axis=1) &
                                           supported_counts[self.counts.astype(int) - 1])
        if not len(self.eligible_leads):
            raise ValueError(f"no compatible lead-analogue pairs in {split}")
        self.max_nodes = self.manifest["max_nodes"]

    def __len__(self):
        return len(self.counts)

    def graph(self, index: int) -> dict:
        return {"atoms": self.atoms[index], "bonds": self.bonds[index],
                "count": int(self.counts[index]), "smiles": self.smiles[index],
                "id": self.ids[index]}

    def sample_episode(self, rng: np.random.Generator, *, support_size: int | None = None,
                       queries_per_lead: int = 4, campaign_index: int | None = None,
                       lead_index: int | None = None) -> EpisodeRecords:
        """Draw requests independently; all query deltas remain response targets."""
        size = int(rng.choice(self.support_sizes)) if support_size is None else int(support_size)
        if size < 1 or size >= len(self):
            raise ValueError("support size must be positive and leave query compounds")
        campaign_index = (int(rng.integers(len(self.campaign_ids)))
                          if campaign_index is None else int(campaign_index))
        lead = int(rng.choice(self.eligible_leads)) if lead_index is None else int(lead_index)
        available = self.neighbours[lead]
        available = available[available >= 0]
        if not len(available) or queries_per_lead < 1:
            raise ValueError("lead requires analogues and a positive query count")
        queries = rng.choice(available, size=queries_per_lead,
                             replace=len(available) < queries_per_lead).astype(np.int64)
        excluded = np.unique(np.r_[queries, lead])
        pool = np.setdiff1d(np.arange(len(self)), excluded, assume_unique=True)
        if size - 1 > len(pool):
            raise ValueError("not enough distinct support compounds outside queries")
        support = np.r_[lead, rng.choice(pool, size=size - 1, replace=False)].astype(np.int64)
        y0 = self.observations[campaign_index, lead].copy()
        y1 = self.observations[campaign_index, queries].copy()
        delta = y1 - y0
        scales = self.manifest["endpoint_scale"]
        goals = draw_goals(rng, y0, len(queries), s1=scales["goal_s1"], s2=scales["goal_s2"])
        weights = feasibility_weights(delta, y0, goals)
        return EpisodeRecords(
            [self.graph(index) for index in support], self.observations[campaign_index, support].copy(),
            self.graph(lead), y0, [self.graph(index) for index in queries], y1, delta,
            goals, weights, self.context.copy(), self.campaign_ids[campaign_index], support,
            lead, queries, self.split,
        )

    def nested_support(self, campaign_index: int, lead_index: int, *, seed: int,
                       sizes=DEFAULT_SUPPORT_SIZES) -> dict:
        """Reserve the full largest support pool at every paired evaluation size."""
        sizes = tuple(sorted(set(int(size) for size in sizes)))
        if min(sizes) < 1 or max(sizes) > len(self):
            raise ValueError("nested support size outside dataset")
        rng = np.random.default_rng(seed)
        others = np.setdiff1d(np.arange(len(self)), [lead_index], assume_unique=True)
        pool = np.r_[lead_index, rng.choice(others, size=max(sizes) - 1, replace=False)].astype(np.int64)
        records = {size: {"graphs": [self.graph(index) for index in pool[:size]],
                          "values": self.observations[campaign_index, pool[:size]].copy(),
                          "context": self.context.copy(), "indices": pool[:size].copy()}
                   for size in sizes}
        return {"records": records, "reserve_indices": pool,
                "reserve_smiles": tuple(self.smiles[index] for index in pool),
                "reserve_ids": tuple(self.ids[index] for index in pool),
                "campaign_id": self.campaign_ids[campaign_index],
                "lead": self.graph(lead_index),
                "lead_values": self.observations[campaign_index, lead_index].copy()}

    def sample_batch(self, rng: np.random.Generator, *, episodes: int = 8,
                     queries_per_lead: int = 4):
        return [self.sample_episode(rng, queries_per_lead=queries_per_lead)
                for _ in range(episodes)]


def collate_episodes(episodes: list[EpisodeRecords], *, pooled_features: torch.Tensor,
                     device="cpu") -> CampaignBatch:
    """Translate observation records to contracts without query-label leakage."""
    if not episodes:
        raise ValueError("cannot collate an empty campaign batch")
    if pooled_features.ndim != 2:
        raise ValueError("pooled_features must index molecular identities by row")
    maximum = max(len(episode.support_indices) for episode in episodes)
    feature_device = pooled_features.device
    features = pooled_features.new_zeros((len(episodes), maximum, pooled_features.shape[1]))
    values = torch.zeros((len(episodes), maximum, 2), dtype=torch.float32, device=feature_device)
    mask = torch.zeros((len(episodes), maximum), dtype=torch.bool, device=feature_device)
    contexts = torch.as_tensor(np.stack([episode.context for episode in episodes]),
                               dtype=torch.float32, device=feature_device)
    query_graphs, lead_graphs, mappings, baselines, query_contexts = [], [], [], [], []
    for episode_index, episode in enumerate(episodes):
        size = len(episode.support_indices)
        indices = torch.as_tensor(episode.support_indices, dtype=torch.long, device=feature_device)
        features[episode_index, :size] = pooled_features.detach().index_select(0, indices)
        values[episode_index, :size] = torch.as_tensor(episode.support_values, device=feature_device)
        mask[episode_index, :size] = True
        number = len(episode.query_graphs)
        query_graphs.extend(episode.query_graphs)
        lead_graphs.extend([episode.lead_graph] * number)
        mappings.extend([episode_index] * number)
        baselines.extend([episode.lead_values] * number)
        query_contexts.extend([episode.context] * number)
    support = SupportBatch(features, values, contexts, mask)
    batch = CampaignBatch(
        collate_graphs(query_graphs), collate_graphs(lead_graphs), support,
        torch.as_tensor(mappings, dtype=torch.long),
        torch.as_tensor(np.stack(baselines), dtype=torch.float32),
        torch.as_tensor(np.stack(query_contexts), dtype=torch.float32),
        torch.as_tensor(np.concatenate([episode.goals for episode in episodes]), dtype=torch.float32),
        torch.as_tensor(np.concatenate([episode.deltas for episode in episodes]), dtype=torch.float32),
        torch.as_tensor(np.concatenate([episode.flow_weights for episode in episodes]), dtype=torch.float32),
    )
    return batch.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", default="2_data_storage/processed/qm9")
    parser.add_argument("--output-dir", default="2_data_storage/processed/calibrated_redesign_toy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-campaigns", type=int, default=32)
    parser.add_argument("--validation-campaigns", type=int, default=8)
    parser.add_argument("--test-campaigns", type=int, default=8)
    parser.add_argument("--molecule-limit", type=int)
    args = parser.parse_args()
    manifest = prepare_campaigns(args.graph_dir, args.output_dir, seed=args.seed,
                                source_campaigns=args.source_campaigns,
                                validation_campaigns=args.validation_campaigns,
                                test_campaigns=args.test_campaigns,
                                molecule_limit=args.molecule_limit)
    print(json.dumps({"splits": manifest["splits"],
                      "unsupported_lead_counts": manifest["unsupported_lead_counts"]}, indent=2))


if __name__ == "__main__":
    main()
