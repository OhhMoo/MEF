"""Operational molecular codec, split, padding, and training-prior checks."""

import numpy as np
import pytest
import torch

from emf.chem import decode_graph, smiles_to_graph
from emf.data import collate_graphs, load_count_prior, load_split, parse_qm9_xyz, prepare_records


@pytest.mark.parametrize("smiles", ["C", "CCO", "C#N", "c1ccccc1", "c1cc[nH]c1", "CC(=O)N", "FCF"])
def test_chemistry_roundtrip(smiles):
    graph = smiles_to_graph(smiles)
    outcome = decode_graph(graph["atoms"], graph["bonds"], graph["count"], 9)
    assert outcome.chemically_valid and outcome.connected
    assert outcome.smiles == graph["smiles"]


@pytest.mark.parametrize("smiles,reason", [
    ("[NH4+]", "formal_charge"), ("CS", "unsupported_element"),
    ("C.C", "disconnected"), ("[CH3]", "radical"), ("[13CH4]", "isotope"),
])
def test_unsupported_chemistry_is_explicit(smiles, reason):
    with pytest.raises(ValueError, match=reason):
        smiles_to_graph(smiles)


def test_disconnected_is_sanitized_but_not_useful():
    outcome = decode_graph(np.array([0, 0]), np.array([0]), 2, 2)
    assert outcome.chemically_valid and not outcome.connected
    assert outcome.smiles == "C.C" and outcome.failure_reason == "disconnected"


def test_invalid_valence_is_preserved_as_failure():
    # Fluorine with two single bonds cannot sanitize.
    outcome = decode_graph(np.array([0, 3, 0]), np.array([1, 0, 1]), 3, 3)
    assert not outcome.chemically_valid and outcome.failure_reason == "invalid_valence"


def test_padding_is_ignored_and_masks_pack_only_active_pairs():
    graph = smiles_to_graph("CO", max_nodes=5)
    graph["atoms"][2:] = 255
    i, j = np.triu_indices(5, k=1)
    graph["bonds"][j >= 2] = 255
    outcome = decode_graph(graph["atoms"], graph["bonds"], 2, 5)
    assert outcome.smiles == "CO"
    graph["id"] = "fixture"
    batch = collate_graphs([graph])
    assert batch.masks.nodes.tolist() == [[True, True, False, False, False]]
    assert batch.masks.pairs.sum().item() == 1
    assert batch.atoms.dtype == torch.long


def test_official_xyz_reads_original_smiles():
    xyz = "5\ngdb9 17 0.1\nC 0 0 0 0\nH 0 0 0 0\nH 0 0 0 0\nH 0 0 0 0\nH 0 0 0 0\n1.2\nC CO\nInChI=1/CH4\n"
    assert parse_qm9_xyz(xyz) == (17, "C")


def test_splits_deduplicate_before_cap_and_fit_prior_on_selected_train(tmp_path):
    smiles = ["C", "N", "O", "F", "CC", "CN", "CO", "CF", "NN", "NO", "OO", "C=C", "C#N", "CCC", "CCO", "CCN", "CCF", "COC", "CNC", "FCF"]
    records = [(str(index), smi) for index, smi in enumerate(smiles)] + [("dup", "OC")]
    manifest = prepare_records(records, tmp_path / "one", train_limit=7, seed=42)
    second = prepare_records(records, tmp_path / "two", train_limit=7, seed=42)
    datasets = {split: load_split(tmp_path / "one", split) for split in ("train", "validation", "test")}
    assert manifest["retained_unique"] == 20
    assert manifest["rejected"]["duplicate_identity"] == 1
    assert len(datasets["train"]) == 7
    identities = {name: set(dataset.smiles) for name, dataset in datasets.items()}
    assert not identities["train"] & identities["validation"]
    assert not identities["train"] & identities["test"]
    assert not identities["validation"] & identities["test"]
    assert manifest["splits"]["train"]["ids"] == second["splits"]["train"]["ids"]
    histogram = np.bincount(datasets["train"].counts, minlength=10)[1:]
    np.testing.assert_allclose(load_count_prior(tmp_path / "one").numpy(), histogram / histogram.sum())
