"""Known distributions and failed attempts exercise offline study analysis."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from emf.chem import smiles_to_graph
from emf.evaluation import write_generation_outputs, summarize_attempts


source = Path(__file__).resolve().parents[1] / "4_experimental_result/control/study_metrics.py"
spec = importlib.util.spec_from_file_location("study_metrics", source)
study_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study_metrics)


def reference_fixture(path):
    path.mkdir()
    graphs = [smiles_to_graph(smiles, max_nodes=3) for smiles in ("CC", "C1CC1")]
    np.savez_compressed(path / "validation.npz",
                        atoms=np.stack([graph["atoms"] for graph in graphs]),
                        bonds=np.stack([graph["bonds"] for graph in graphs]),
                        counts=[graph["count"] for graph in graphs],
                        smiles=[graph["smiles"] for graph in graphs], ids=["1", "2"])
    # Analysis must not open the reserved test split or consult a training split.
    (path / "test.npz").write_text("reserved test split is deliberately unreadable")
    return graphs


def attempt(graph, *, novel=True):
    return {"atom_count": graph["count"], "atoms": graph["atoms"].tolist(),
            "bonds": graph["bonds"].tolist(), "smiles": graph["smiles"],
            "chemically_valid": True, "connected": True,
            "failure_reason": None, "novel": novel}


def test_quality_comparisons_preserve_failures_and_exclude_empty_scaffolds(tmp_path):
    reference = tmp_path / "reference"
    graphs = reference_fixture(reference)
    output = tmp_path / "generation"
    ledger = [attempt(graphs[0], novel=False), attempt(graphs[1]),
              {"chemically_valid": True, "connected": False, "smiles": "C.C",
               "novel": None, "failure_reason": "disconnected"},
              {"chemically_valid": False, "connected": False, "smiles": None,
               "novel": None, "failure_reason": "invalid_valence"}]
    write_generation_outputs(output, ledger, summarize_attempts(ledger, 10.0))
    original_ledger = (output / "attempts.jsonl").read_bytes()
    metrics = study_metrics.analyze_samples(output, reference)

    assert metrics["attempts"] == 4
    assert metrics["connected_valid"] == metrics["distinct_connected_valid"] == 2
    assert metrics["novel_distinct_connected_valid"] == 1
    assert metrics["connected_validity_rate"] == 0.5
    assert metrics["distinct_connected_valid_per_second"] == 0.2
    assert metrics["failure_disconnected_count"] == metrics["failure_invalid_valence_count"] == 1
    assert metrics["nonempty_scaffold_attempts"] == metrics["distinct_nonempty_scaffolds"] == 1
    assert metrics["scaffold_diversity_among_nonempty"] == 1.0
    assert metrics["acyclic_fraction_among_connected_valid"] == 0.5
    assert metrics["heldout_atom_label_js_nats"] == metrics["heldout_bond_label_js_nats"] == 0.0
    for descriptor in study_metrics.DESCRIPTORS:
        assert metrics[f"heldout_{descriptor}_wasserstein_1"] == 0.0
    assert metrics["heldout_reference_split"] == "validation"
    assert metrics["heldout_reference_sha256"] == hashlib.sha256((reference / "validation.npz").read_bytes()).hexdigest()
    assert (output / "attempts.jsonl").read_bytes() == original_ledger
    assert all(value is None or isinstance(value, (str, int, float)) for value in metrics.values())
    assert json.loads((output / "quality_metrics.json").read_text()) == metrics
    with (output / "quality_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["attempts"] == "4"


def test_no_valid_molecules_yields_null_conditional_distances(tmp_path):
    reference = tmp_path / "reference"
    reference_fixture(reference)
    output = tmp_path / "generation"
    ledger = [{"chemically_valid": False, "connected": False, "smiles": None,
               "novel": None, "failure_reason": "invalid_valence"}]
    write_generation_outputs(output, ledger, summarize_attempts(ledger, 2.0))
    metrics = study_metrics.analyze_samples(output, reference)
    assert metrics["attempts"] == 1 and metrics["connected_valid"] == 0
    assert metrics["heldout_reference_molecules"] == 2
    for name, value in metrics.items():
        if name.endswith(("_js_nats", "_wasserstein_1")):
            assert value is None
    assert metrics["scaffold_diversity_among_nonempty"] is None
    assert metrics["acyclic_fraction_among_connected_valid"] is None
    assert metrics["connected_valid_atom_fraction_C"] is None
    assert metrics["distinct_connected_valid_per_second"] == 0.0


def test_empirical_distances_have_their_known_exact_values():
    assert study_metrics._wasserstein_1([0.0, 2.0], [1.0]) == 1.0
    assert study_metrics._wasserstein_1([0.0, 0.0, 2.0], [0.0, 2.0, 2.0]) == pytest.approx(2 / 3)
    assert study_metrics._wasserstein_1([], [1.0]) is None
    assert study_metrics._js_nats([1, 0], [0, 1]) == pytest.approx(math.log(2))
    assert study_metrics._js_nats([1, 2], [2, 4]) == 0.0
    atoms, bonds = study_metrics._label_counts([{"atoms": [0], "bonds": [], "count": 1}])
    assert atoms.tolist() == [1, 0, 0, 0]
    assert bonds.sum() == 0
