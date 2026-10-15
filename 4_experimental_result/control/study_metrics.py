"""Compare preserved generation attempts with the QM9 validation split.

Validity and useful yield include every attempt. Label and descriptor comparisons
are conditional on connected valid attempts, with repetitions retained. The test
split is never opened. No generation time is attributed to this offline analysis.
"""

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from emf.contracts import ATOM_TYPES, BOND_TYPES
from emf.data import load_split
from emf.evaluation import summarize_attempts


DESCRIPTORS = {
    "molwt": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "ring_count": rdMolDescriptors.CalcNumRings,
}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _wasserstein_1(left, right):
    """Exact empirical one-dimensional W1: integral of |F_left - F_right|."""
    left, right = np.sort(left), np.sort(right)
    if not len(left) or not len(right):
        return None
    support = np.unique(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, support[:-1], side="right") / len(left)
    right_cdf = np.searchsorted(right, support[:-1], side="right") / len(right)
    return float(np.dot(np.abs(left_cdf - right_cdf), np.diff(support)))


def _js_nats(left, right):
    """Jensen-Shannon divergence of categorical frequencies, in natural units."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.sum() == 0 or right.sum() == 0:
        return None
    left, right = left / left.sum(), right / right.sum()
    midpoint = (left + right) / 2
    result = 0.0
    for distribution in (left, right):
        active = distribution > 0
        result += np.dot(distribution[active], np.log(distribution[active] / midpoint[active])) / 2
    return float(result)


def _label_counts(graphs):
    """Pool active atoms and active unordered pairs, including the no-bond label."""
    atoms, bonds = np.zeros(len(ATOM_TYPES)), np.zeros(len(BOND_TYPES))
    for graph in graphs:
        count = int(graph.get("atom_count", graph.get("count", 0)))
        node_labels, pair_labels = np.asarray(graph["atoms"]), np.asarray(graph["bonds"])
        max_nodes = len(node_labels)
        if not 1 <= count <= max_nodes or len(pair_labels) != max_nodes * (max_nodes - 1) // 2:
            raise ValueError("connected valid graph has inconsistent dimensions")
        _, right = np.triu_indices(max_nodes, k=1)
        node_labels, pair_labels = node_labels[:count], pair_labels[right < count]
        for labels, histogram in ((node_labels, atoms), (pair_labels, bonds)):
            if labels.size and (labels.dtype.kind not in "iu" or np.any(labels < 0)
                                or np.any(labels >= len(histogram))):
                raise ValueError("connected valid graph has an unsupported category")
            histogram += np.bincount(labels.astype(int), minlength=len(histogram))
    return atoms, bonds


def _molecules(smiles):
    molecules = []
    for identity in smiles:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(identity) if isinstance(identity, str) else None
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("a connected valid identity cannot be parsed as one molecule")
        molecules.append(molecule)
    return molecules


def _descriptor_values(molecules):
    return {name: np.array([function(molecule) for molecule in molecules], dtype=float)
            for name, function in DESCRIPTORS.items()}


@lru_cache(maxsize=4)
def _reference_statistics(validation_path, digest):
    # The digest keys immutable contents, so repeated NFE comparisons reuse work.
    reference = load_split(Path(validation_path).parent, "validation")
    molecules = _molecules(reference.smiles)
    atom_counts, bond_counts = _label_counts(reference[index] for index in range(len(reference)))
    return len(reference), atom_counts, bond_counts, _descriptor_values(molecules)


def analyze_samples(output_dir, reference_data_dir):
    """Return flat scalar/null metrics and save quality_metrics.json and .csv.

    Murcko diversity divides distinct nonempty scaffolds by nonempty-scaffold
    connected valid attempts. Acyclic molecules are counted separately. JS pools
    active categories; W1 uses equally weighted molecule descriptors in their
    original units. Missing conditional populations yield null, never zero.
    """
    output_dir = Path(output_dir)
    ledger_path = output_dir / "attempts.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    summary = json.loads((output_dir / "summary.json").read_text())
    metrics = summarize_attempts(ledger, summary["generation_seconds"])
    failures = metrics.pop("failure_counts")
    metrics.update({f"failure_{reason}_count": count for reason, count in failures.items()})
    for name in ("seed", "nfe", "T", "batch_size", "device"):
        if name in summary:
            metrics[name] = summary[name]
    metadata = summary.get("metadata", {})
    metrics["checkpoint_sha256"] = metadata.get("checkpoint_sha256")

    validation_path = (Path(reference_data_dir) / "validation.npz").resolve(strict=True)
    reference_hash = _sha256(validation_path)
    reference_count, reference_atoms, reference_bonds, reference_descriptors = _reference_statistics(
        str(validation_path), reference_hash)
    valid = [row for row in ledger if row["chemically_valid"] and row["connected"]]
    molecules = _molecules(row["smiles"] for row in valid)
    atoms, bonds = _label_counts(valid)
    descriptors = _descriptor_values(molecules)
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
                 for molecule in molecules]
    nonempty = [scaffold for scaffold in scaffolds if scaffold]
    metrics.update({
        "distribution_scope": "connected_valid_attempts_with_repetitions",
        "atom_frequency_scope": "pooled_active_atoms",
        "bond_frequency_scope": "pooled_active_unordered_pairs_including_none",
        "descriptor_distance_units": "MolWt:g/mol; LogP:dimensionless; TPSA:angstrom^2; ring_count:count",
        "heldout_reference_split": "validation",
        "heldout_reference_path": str(validation_path),
        "heldout_reference_sha256": reference_hash,
        "heldout_reference_molecules": reference_count,
        "attempt_ledger_sha256": _sha256(ledger_path),
        "analysis_rdkit_version": rdBase.rdkitVersion,
        "nonempty_scaffold_attempts": len(nonempty),
        "distinct_nonempty_scaffolds": len(set(nonempty)),
        "scaffold_diversity_among_nonempty": len(set(nonempty)) / len(nonempty) if nonempty else None,
        "acyclic_fraction_among_connected_valid": (len(valid) - len(nonempty)) / len(valid) if valid else None,
        "heldout_atom_label_js_nats": _js_nats(atoms, reference_atoms),
        "heldout_bond_label_js_nats": _js_nats(bonds, reference_bonds),
    })
    for prefix, labels, generated, reference in (
        ("atom", ATOM_TYPES, atoms, reference_atoms),
        ("bond", BOND_TYPES, bonds, reference_bonds),
    ):
        for index, label in enumerate(labels):
            metrics[f"connected_valid_{prefix}_fraction_{label}"] = float(generated[index] / generated.sum()) if generated.sum() else None
            metrics[f"heldout_{prefix}_fraction_{label}"] = float(reference[index] / reference.sum()) if reference.sum() else None
    for name in DESCRIPTORS:
        generated, reference = descriptors[name], reference_descriptors[name]
        metrics[f"heldout_{name}_wasserstein_1"] = _wasserstein_1(generated, reference)
        metrics[f"connected_valid_{name}_mean"] = float(generated.mean()) if len(generated) else None
        metrics[f"heldout_{name}_mean"] = float(reference.mean()) if len(reference) else None

    (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    with (output_dir / "quality_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-data-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_samples(args.output_dir, args.reference_data_dir), indent=2))


if __name__ == "__main__":
    main()
