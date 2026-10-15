"""Small checks for masked chemistry, split discipline, and exact scoring."""

import numpy as np

from emf.masked_edit_data import (MaskedEditDataset, prepare_masked_edit_data,
                                  fit_support_response, UPPER)
from emf.masked_edit_evaluation import ExactMaskedEvaluator


def test_molecule_splits_masks_and_episodes(tmp_path):
    directory = tmp_path / "masked"
    manifest = prepare_masked_edit_data(directory)
    splits = {name: set(indices) for name, indices in manifest["molecule_splits"].items()}
    assert not (splits["train"] & splits["validation"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["validation"] & splits["test"])
    assert len(set.union(*splits.values())) == 294
    assert {name: record["compatible_count"] for name, record in manifest["masks"].items()} == {
        "f0_left": 25, "f0_right": 126, "f1_left": 69, "f1_right": 54}

    for split in ("train", "validation", "test"):
        dataset = MaskedEditDataset(directory, split, kind="vector")
        rng = np.random.default_rng(10)
        for campaign_index, cid in enumerate(dataset.campaign_ids):
            episode = dataset.sample_episode(rng, support_size=4, queries_per_lead=2,
                                             campaign_index=campaign_index)
            assert set(episode.support_indices).isdisjoint(episode.query_indices)
            assert set(episode.query_indices) <= splits[split]
            assert set(episode.support_indices) - {episode.lead_index} <= splits[split]
            assert all(np.array_equal(graph["atoms"][~episode.atom_editable],
                                      episode.lead_graph["atoms"][~episode.atom_editable])
                       for graph in episode.query_graphs)
            assert all(np.array_equal(graph["bonds"][~episode.bond_editable],
                                      episode.lead_graph["bonds"][~episode.bond_editable])
                       for graph in episode.query_graphs)
            assert np.all(np.isfinite(dataset.observations[cid][episode.support_indices]))
            withheld = set.union(*(splits[name] for name in splits if name != split)) - {episode.lead_index}
            assert np.all(np.isnan(dataset.observations[cid][list(withheld)]))
            reserve = dataset.nested_support(campaign_index, seed=42, sizes=(4, 8, 16))
            assert len(reserve["records"][16]["indices"]) == 16
            assert set(reserve["records"][4]["indices"]) <= set(reserve["records"][16]["indices"])
            assert set(reserve["query_indices"]).isdisjoint(reserve["reserve_indices"])

    source = MaskedEditDataset(directory, "train", kind="vector")
    pairs = source.sample_pairs(np.random.default_rng(11), 32)
    assert len(pairs) == 32
    assert all(pair["query_index"] in splits["train"] for pair in pairs)
    assert all(pair["bond_editable"].shape == UPPER[0].shape for pair in pairs)
    reserve = source.nested_support(0, seed=13, sizes=(4, 8, 16))
    record = reserve["records"][16]
    fit = fit_support_response([graph["smiles"] for graph in record["graphs"]],
                               record["values"], reserve["lead"]["smiles"],
                               source.manifest["descriptor_scales_from_source_molecules"])
    assert fit["coefficients"].shape == (2, 3)
    assert 0 <= fit["basis_rank"] <= 3


def test_exact_evaluator_commits_before_oracle_and_tracks_uniqueness(tmp_path):
    directory = tmp_path / "masked"
    prepare_masked_edit_data(directory)
    dataset = MaskedEditDataset(directory, "validation", kind="vector")
    evaluator = ExactMaskedEvaluator(directory / "private" / "oracle.json")
    cid = dataset.campaign_ids[0]
    meta = dataset.manifest["campaigns"][cid]
    mask = dataset.masks[meta["mask_id"]]
    index = meta["molecule_indices"][0]
    graph = dataset.graph(index, aligned=True, task_id=mask.mask_id)
    first = evaluator.ledger.commit(graph, mask)
    assert first.protected_copy and first.global_unique
    result = evaluator.evaluate_committed(first, cid)
    assert result["chemically_valid"] and result["connected"]
    assert result["atom_edits"] + result["bond_edits"] > 0
    assert result["exact_delta"] is not None
    assert len(evaluator.ledger.attempts) == 1
    second = evaluator.ledger.commit(graph, mask)
    assert not second.global_unique

    invalid_scope = {**graph, "atoms": graph["atoms"].copy()}
    protected_index = int(np.flatnonzero(~mask.atom_editable)[0])
    invalid_scope["atoms"][protected_index] = (int(invalid_scope["atoms"][protected_index]) + 1) % 4
    third = evaluator.ledger.commit(invalid_scope, mask)
    assert not third.protected_copy
    assert evaluator.evaluate_committed(third, cid)["exact_delta"] is None
