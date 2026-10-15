"""Evidence-steering chemistry, paired observations, and episode isolation."""

import json

import numpy as np
import pytest
import torch
from rdkit import DataStructs

from emf.campaigns import collate_episodes, molecular_fingerprint
from emf.chem import smiles_to_graph
from emf.steering_data import (DESCRIPTORS, GOAL, MIN_SIMILARITY, SteeringDataset,
                               SteeringOracle, prepare_steering_data)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("steering_data")
    manifest = prepare_steering_data(directory)
    return directory, manifest


def test_declared_families_are_valid_local_and_large(prepared):
    directory, manifest = prepared
    dataset = SteeringDataset(directory)
    assert len(dataset.family_leads) == 2
    assert all(size >= 128 for size in manifest["family_sizes"])
    assert len(dataset.smiles) == len(set(dataset.smiles))
    for family, lead in zip(dataset.family_indices, dataset.family_leads):
        reference = dataset.smiles[lead]
        fingerprint = molecular_fingerprint(reference)
        count = dataset.counts[lead]
        for index in family:
            graph = smiles_to_graph(dataset.smiles[index])
            assert graph["smiles"] == dataset.smiles[index]
            assert graph["count"] == count
            similarity = DataStructs.TanimotoSimilarity(
                fingerprint, molecular_fingerprint(dataset.smiles[index]))
            assert similarity >= MIN_SIMILARITY


def test_paired_campaigns_share_graphs_support_and_baseline(prepared):
    directory, _ = prepared
    dataset = SteeringDataset(directory, "test")
    oracle = SteeringOracle.load(directory / "private" / "oracle.json")
    for first_index in range(0, len(dataset.campaign_ids), 2):
        first, second = dataset.campaign_ids[first_index:first_index + 2]
        assert dataset.campaign_pairs[first] == second
        assert dataset.campaign_family[first] == dataset.campaign_family[second]
        lead = dataset.campaign_lead_indices[first]
        assert lead == dataset.campaign_lead_indices[second]
        np.testing.assert_array_equal(dataset.observations[first_index, lead], [0, 0])
        np.testing.assert_array_equal(dataset.observations[first_index + 1, lead], [0, 0])
        np.testing.assert_allclose(dataset.observations[first_index, :, 0],
                                   -dataset.observations[first_index + 1, :, 0], atol=1e-7)
        np.testing.assert_array_equal(dataset.observations[first_index, :, 1],
                                      dataset.observations[first_index + 1, :, 1])
        seed = dataset.paired_support_seed(first)
        assert seed == dataset.paired_support_seed(second)
        a = dataset.nested_support(first_index, lead, seed=seed)
        b = dataset.nested_support(first_index + 1, lead, seed=seed)
        assert a["reserve_ids"] == b["reserve_ids"]
        assert a["reserve_smiles"] == b["reserve_smiles"]
        for size in dataset.support_sizes:
            np.testing.assert_array_equal(a["records"][size]["indices"],
                                          b["records"][size]["indices"])
        assert oracle.coefficients[first] == -oracle.coefficients[second]


def test_episode_has_family_local_disjoint_queries_and_fixed_goal(prepared):
    directory, _ = prepared
    dataset = SteeringDataset(directory)
    episode = dataset.sample_episode(np.random.default_rng(31), support_size=64,
                                     queries_per_lead=16, campaign_index=1)
    family = dataset.campaign_family[episode.campaign_id]
    assert episode.lead_index == dataset.family_leads[family]
    assert episode.support_indices[0] == episode.lead_index
    assert len(set(episode.support_indices)) == 64
    assert not set(episode.support_indices) & set(episode.query_indices)
    assert set(episode.support_indices) | set(episode.query_indices) <= set(dataset.family_indices[family])
    np.testing.assert_array_equal(episode.goals, np.repeat(GOAL[None], 16, axis=0))
    np.testing.assert_array_equal(episode.flow_weights,
        ((episode.deltas[:, 0] >= GOAL[0]) & (episode.deltas[:, 1] >= GOAL[1])).astype(np.float32))
    assert not hasattr(episode, "coefficient")
    assert episode.context.shape == (4,)


def test_existing_collator_accepts_mixed_family_episodes(prepared):
    directory, _ = prepared
    dataset = SteeringDataset(directory)
    episodes = [dataset.sample_episode(np.random.default_rng(20 + family),
                                       support_size=size, campaign_index=family * 32)
                for family, size in ((0, 4), (1, 64))]
    pooled = torch.randn(len(dataset), 7)
    batch = collate_episodes(episodes, pooled_features=pooled)
    assert batch.support.mask.sum(axis=1).tolist() == [4, 64]
    assert batch.goal.shape == (8, 2)
    torch.testing.assert_close(batch.goal, torch.tensor(GOAL).repeat(8, 1))
    for row, episode in enumerate(episodes):
        indices = torch.as_tensor(episode.support_indices)
        torch.testing.assert_close(batch.support.features[row, :len(indices)], pooled[indices])
    assert not hasattr(batch.support, "campaign_id")


def test_private_oracle_scores_exact_library_and_valid_novel_graph(prepared):
    directory, manifest = prepared
    dataset = SteeringDataset(directory, "validation")
    oracle = SteeringOracle.load(directory / "private" / "oracle.json")
    assert manifest["descriptor_definition"] == DESCRIPTORS
    for campaign_index, campaign_id in enumerate(dataset.campaign_ids):
        for index in (0, 20, len(dataset) - 1):
            np.testing.assert_allclose(oracle.score_smiles(campaign_id, dataset.smiles[index]),
                                       dataset.observations[campaign_index, index], atol=1e-6)
    values = oracle.score_smiles(dataset.campaign_ids[0], "CCN(CC)CCCO")
    assert values.shape == (2,) and np.isfinite(values).all()
    with pytest.raises(ValueError):
        oracle.score_smiles(dataset.campaign_ids[0], "CCS")


def test_all_case_feasibility_and_fresh_heldout_coefficients(prepared):
    directory, manifest = prepared
    report = json.loads((directory / "feasibility.json").read_text())
    oracle = SteeringOracle.load(directory / "private" / "oracle.json")
    assert len(report["cases"]) == sum(len(manifest["splits"][split]["campaign_ids"])
                                        for split in ("train", "validation", "test"))
    datasets = {split: SteeringDataset(directory, split)
                for split in ("train", "validation", "test")}
    for case in report["cases"]:
        dataset = datasets[case["split"]]
        campaign = dataset.campaign_ids.index(case["campaign_id"])
        family = dataset.family_indices[case["family"]]
        values = dataset.observations[campaign]
        feasible = [index for index in family
                    if values[index, 0] >= GOAL[0] and values[index, 1] >= GOAL[1]]
        assert case["feasible_library"] == len(feasible)
        assert case["feasible_outside_max_support"] == len(set(feasible) - set(case["reserve_indices"]))
        assert case["support_seed"] == dataset.paired_support_seed(case["campaign_id"])
    pilot = json.loads((directory / "construction_pilot.json").read_text()) if (
        directory / "construction_pilot.json").exists() else None
    if pilot is not None:
        assert pilot["status"] == "discarded_task_construction_pilot"
    old = {0.6, 1.0, 1.4, 1.8, 0.8, 1.6}
    heldout = {abs(value) for key, value in oracle.coefficients.items()
               if key.startswith(("validation_", "test_"))}
    assert not heldout & old
    for case in report["cases"]:
        if case["split"] == "test":
            assert case["feasible_outside_max_support"] > 0
