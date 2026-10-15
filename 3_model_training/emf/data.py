"""Prepare the authors' QM9 SMILES and load compact masked graph batches.

Raw archives, exact molecule-disjoint splits, and source hashes are persisted.
The training cap applies after an 80/10/10 split; the count prior is train-only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Iterable
import urllib.request

import numpy as np
from rdkit import rdBase
import torch
from torch.utils.data import Dataset

from .chem import decode_graph, smiles_to_graph
from .contracts import GraphBatch, GraphMasks

QM9_SOURCES = {
    "dsgdb9nsd.xyz.tar.bz2": {
        "url": "https://ndownloader.figshare.com/files/3195389",
        "md5": "ad1ebd51ee7f5b3a6e32e974e5d54012", "size": 86144227},
    "uncharacterized.txt": {
        "url": "https://ndownloader.figshare.com/files/3195404",
        "md5": "a361887bacb427b8a0ce7903d92a53b4", "size": 486752},
    "readme.txt": {
        "url": "https://ndownloader.figshare.com/files/3195392",
        "md5": "c6581a03f673746528c57acfc6b79679", "size": 5198},
}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_qm9(raw_dir) -> dict:
    """Download immutable file IDs and verify authors' published integrity hashes."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, expected in QM9_SOURCES.items():
        target = raw_dir / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(expected["url"], headers={"User-Agent": "EMF-QM9/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as sink:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        sink.write(block)
                if (temporary.stat().st_size != expected["size"] or
                        _digest(temporary, "md5") != expected["md5"]):
                    raise ValueError(f"QM9 source integrity check failed: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if target.stat().st_size != expected["size"] or _digest(target, "md5") != expected["md5"]:
            raise ValueError(f"QM9 source integrity check failed: {target}")
        sources[name] = {**expected, "sha256": _digest(target)}
    return sources


def parse_qm9_xyz(text: str) -> tuple[int, str]:
    """Read the one-based ID and original GDB9 SMILES, not relaxed geometry."""
    lines = text.splitlines()
    atom_count = int(lines[0])
    molecule_id = int(lines[1].split()[1])
    smiles = lines[atom_count + 3].split()[0]
    return molecule_id, smiles


def _read_exclusions(path: Path) -> set[int]:
    ids = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            ids.add(int(fields[0]))
    if len(ids) != 3054:
        raise ValueError(f"Expected 3054 original QM9 exclusions, found {len(ids)}")
    return ids


def _qm9_records(archive: Path, exclusions: set[int]) -> Iterable[tuple[str, str]]:
    # Stream tar members; never extract paths or repeatedly seek compressed data.
    with tarfile.open(archive, mode="r|bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".xyz"):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read molecular archive member: {member.name}")
            with stream:
                molecule_id, smiles = parse_qm9_xyz(stream.read().decode("utf-8"))
            if molecule_id not in exclusions:
                yield f"qm9:{molecule_id}", smiles


def prepare_records(records: Iterable[tuple[str, str]], output_dir, *,
                    train_limit: int | None = 20000, seed: int = 42,
                    max_nodes: int = 9, provenance: dict | None = None) -> dict:
    """Standardize/deduplicate identities before reproducible split selection."""
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive or None")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_identity, rejected = {}, Counter()
    total = 0
    for source_id, smiles in records:
        total += 1
        try:
            graph = smiles_to_graph(smiles, max_nodes)
            outcome = decode_graph(graph["atoms"], graph["bonds"], graph["count"], max_nodes)
            if not outcome.chemically_valid or not outcome.connected or outcome.smiles != graph["smiles"]:
                raise ValueError("codec_roundtrip")
        except (ValueError, RuntimeError) as error:
            rejected[str(error)] += 1
            continue
        identity = graph["smiles"]
        if identity in by_identity:
            rejected["duplicate_identity"] += 1
            continue
        by_identity[identity] = {**graph, "id": str(source_id)}

    if len(by_identity) < 10:
        raise ValueError("Need at least ten unique supported molecules for 80/10/10 splits")
    graphs = [by_identity[identity] for identity in sorted(by_identity)]
    selected = np.random.default_rng(seed).permutation(len(graphs))
    train_end, valid_end = int(0.8 * len(graphs)), int(0.9 * len(graphs))
    indices = {"train": selected[:train_end], "validation": selected[train_end:valid_end],
               "test": selected[valid_end:]}
    if train_limit is not None:
        indices["train"] = indices["train"][:train_limit]

    split_info = {}
    for split, positions in indices.items():
        subset = [graphs[int(index)] for index in positions]
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, atoms=np.stack([g["atoms"] for g in subset]),
                            bonds=np.stack([g["bonds"] for g in subset]),
                            counts=np.array([g["count"] for g in subset], dtype=np.uint8),
                            smiles=np.array([g["smiles"] for g in subset]),
                            ids=np.array([g["id"] for g in subset]))
        split_info[split] = {"size": len(subset), "file": path.name,
                             "sha256": _digest(path),
                             "smiles": [g["smiles"] for g in subset],
                             "ids": [g["id"] for g in subset]}

    train_counts = np.array([graphs[int(index)]["count"] for index in indices["train"]])
    histogram = np.bincount(train_counts, minlength=max_nodes + 1)[1:max_nodes + 1]
    count_prior = (histogram / histogram.sum()).tolist()
    identity_hash = hashlib.sha256("\n".join(sorted(by_identity)).encode()).hexdigest()
    manifest = {"schema_version": 1, "dataset": "restricted_qm9_2d",
                "split_recipe": "canonical identity dedup, seeded permutation, 80/10/10, then train cap",
                "seed": seed, "max_nodes": max_nodes, "train_limit": train_limit,
                "rdkit_version": rdBase.rdkitVersion, "source_records": total,
                "retained_unique": len(graphs), "rejected": dict(sorted(rejected.items())),
                "identity_sha256": identity_hash, "sources": provenance or {},
                "count_prior": count_prior, "count_histogram": histogram.tolist(),
                "splits": split_info}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_qm9(output_dir, raw_dir, train_limit: int | None = 20000,
                seed: int = 42, max_nodes: int = 9) -> dict:
    """Acquire official sources and prepare a reproducible exploratory dataset."""
    sources = download_qm9(raw_dir)
    raw_dir = Path(raw_dir)
    exclusions = _read_exclusions(raw_dir / "uncharacterized.txt")
    return prepare_records(_qm9_records(raw_dir / "dsgdb9nsd.xyz.tar.bz2", exclusions),
                           output_dir, train_limit=train_limit, seed=seed,
                           max_nodes=max_nodes, provenance=sources)


class GraphDataset(Dataset):
    """Read-only compact split; labels are converted to int64 at collation."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as arrays:
            self.atoms, self.bonds = arrays["atoms"], arrays["bonds"]
            self.counts = arrays["counts"]
            self.smiles, self.ids = tuple(arrays["smiles"].tolist()), tuple(arrays["ids"].tolist())

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        return {"atoms": self.atoms[index], "bonds": self.bonds[index],
                "count": int(self.counts[index]), "smiles": self.smiles[index], "id": self.ids[index]}


def load_split(processed_dir, split: str) -> GraphDataset:
    if split == "val":
        split = "validation"
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    return GraphDataset(Path(processed_dir) / f"{split}.npz")


def collate_graphs(samples: list[dict]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    atoms = torch.tensor(np.stack([sample["atoms"] for sample in samples]), dtype=torch.long)
    bonds = torch.tensor(np.stack([sample["bonds"] for sample in samples]), dtype=torch.long)
    counts = torch.tensor([sample["count"] for sample in samples], dtype=torch.long)
    masks = GraphMasks.from_counts(counts, atoms.shape[1])
    return GraphBatch(atoms, bonds, masks,
                      tuple(sample["smiles"] for sample in samples),
                      tuple(sample["id"] for sample in samples))


def load_count_prior(processed_dir) -> torch.Tensor:
    """Probability vector indexed by count minus one, fitted on selected training."""
    manifest = json.loads((Path(processed_dir) / "manifest.json").read_text())
    return torch.tensor(manifest["count_prior"], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="2_data_storage/raw/qm9")
    parser.add_argument("--output-dir", default="2_data_storage/processed/qm9")
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=9)
    args = parser.parse_args()
    manifest = prepare_qm9(args.output_dir, args.raw_dir,
                          train_limit=args.train_limit or None, seed=args.seed, max_nodes=args.max_nodes)
    print(json.dumps({"retained_unique": manifest["retained_unique"],
                      "split_sizes": {name: item["size"] for name, item in manifest["splits"].items()},
                      "rejected": manifest["rejected"], "count_prior": manifest["count_prior"]}, indent=2))


if __name__ == "__main__":
    main()
