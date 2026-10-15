"""Campaign isolation, independent requests, exact truth, and count-law checks."""

from itertools import product

import numpy as np
import pytest
import torch

from emf.campaigns import (CampaignDataset, collate_episodes, conditional_count_prior, draw_goals,
                           feasibility_weights, neighbourhood, prepare_campaigns)
from emf.chem import smiles_to_graph
from emf.data import prepare_records
from emf.toy_oracle import ToyOracle, structural_features


@pytest.fixture(scope="module")
def campaign_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("redesign_campaigns")
    identities = set()
    for length in range(2, 7):
        for atoms in product("CNO", repeat=length):
            try:
                identity = smiles_to_graph("".join(atoms))["smiles"]
            except ValueError:
                continue
            identities.add(identity)
            if len(identities) >= 400:
                break
        if len(identities) >= 400:
            break
    records = [(f"fixture:{index}", smiles) for index, smiles in enumerate(sorted(identities))]
    prepare_records(records, directory / "graphs", train_limit=None, seed=7)
    prepare_campaigns(directory / "graphs", directory / "campaigns", seed=13,
                      source_campaigns=4, validation_campaigns=2, test_campaigns=2,
                      support_sizes=(4, 8), min_similarity=0.1)
    return directory / "campaigns"


def test_campaign_and_molecule_splits_are_disjoint(campaign_dir):
    datasets = [CampaignDataset(campaign_dir, split) for split in ("train", "validation", "test")]
    for i, left in enumerate(datasets):
        for right in datasets[i + 1:]:
            assert not set(left.smiles) & set(right.smiles)
            assert not set(left.campaign_ids) & set(right.campaign_ids)
    source = datasets[0]
    np.testing.assert_allclose(source.observations.mean(axis=(0, 1)), 0, atol=1e-6)
    np.testing.assert_allclose(source.observations.std(axis=(0, 1)), 1, atol=1e-6)


def test_query_outcomes_are_absent_from_support_and_do_not_set_goals(campaign_dir):
    data = CampaignDataset(campaign_dir)
    lead = int(data.eligible_leads[0])
    first = data.sample_episode(np.random.default_rng(9), support_size=8,
                                campaign_index=0, lead_index=lead)
    assert len(set(first.support_indices)) == 8
    assert first.support_indices[0] == lead
    assert not set(first.support_indices) & set(first.query_indices)
    for query in np.unique(first.query_indices):
        data.observations[0, query] -= 100
    second = data.sample_episode(np.random.default_rng(9), support_size=8,
                                 campaign_index=0, lead_index=lead)
    np.testing.assert_array_equal(first.goals, second.goals)
    np.testing.assert_array_equal(first.support_values, second.support_values)
    assert len(second.deltas) == 4  # Negative modifications remain response targets.
    assert not second.flow_weights.any()
    np.testing.assert_array_equal(second.flow_weights,
                                  feasibility_weights(second.deltas, second.lead_values, second.goals))


def test_nested_support_reserves_the_largest_pool_at_every_size(campaign_dir):
    data = CampaignDataset(campaign_dir, "test")
    lead = int(data.eligible_leads[0])
    nested = data.nested_support(0, lead, seed=100, sizes=(4, 8))
    assert len(nested["reserve_indices"]) == len(set(nested["reserve_smiles"])) == 8
    np.testing.assert_array_equal(nested["records"][4]["indices"], nested["reserve_indices"][:4])
    np.testing.assert_array_equal(nested["records"][8]["indices"], nested["reserve_indices"])
    assert all(record["indices"][0] == lead for record in nested["records"].values())


def test_count_prior_uses_all_source_pairs_without_response_filtering(campaign_dir):
    data = CampaignDataset(campaign_dir)
    prior, histogram, unsupported = conditional_count_prior(data.counts, data.neighbours)
    np.testing.assert_allclose(prior, data.count_prior)
    np.testing.assert_allclose(prior.sum(axis=1), histogram.sum(axis=1) > 0)
    assert histogram.sum() == (data.neighbours >= 0).sum()
    for count in unsupported:
        assert prior[count - 1].sum() == 0
    data.observations.fill(-999)
    unchanged, _, _ = conditional_count_prior(data.counts, data.neighbours)
    np.testing.assert_array_equal(prior, unchanged)


def test_oracle_evaluates_novel_graphs_reproducibly_and_checks_chemistry(campaign_dir):
    oracle = ToyOracle.load(campaign_dir / "private" / "oracle.json")
    # Fluorinated graph is absent from the C/N/O fixture but supported by the encoder.
    values = oracle.score_smiles("test_000", "CCCF")
    assert values.shape == (2,) and np.isfinite(values).all()
    np.testing.assert_array_equal(values, oracle.score_smiles("test_000", "FCCC"))
    with pytest.raises(ValueError, match="unsupported_element"):
        oracle.score_smiles("test_000", "CS")
    with pytest.raises(ValueError, match="disconnected"):
        oracle.score_smiles("test_000", "C.C")
    features = np.stack([structural_features(s) for s in ("CCO", "CCN", "CCCF")])
    np.testing.assert_allclose(oracle.score_features("test_000", features),
                               np.stack([oracle.score_smiles("test_000", s)
                                         for s in ("CCO", "CCN", "CCCF")]))


def test_goal_prior_keeps_preservation_threshold_below_the_lead():
    goals = draw_goals(np.random.default_rng(17), np.asarray([0.3, -0.6]), 20000)
    assert (goals[:, 1] <= -0.6).all() and (goals[:, 1] >= -1.6).all()
    assert goals[:, 0].min() >= -1 and goals[:, 0].max() <= 1
    assert abs(np.corrcoef(goals.T)[0, 1]) < 0.03


def test_neighbourhood_rule_does_not_use_endpoints():
    same = neighbourhood("CCO", "OCC")
    assert same["similarity"] == 1 and same["count_difference"] == 0 and same["eligible"]
    distant = neighbourhood("CC", "CCCCCCCC", min_similarity=0)
    assert distant["count_difference"] == 6 and not distant["eligible"]


def test_public_loader_does_not_require_private_coefficients(campaign_dir, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("training loader accessed private oracle")
    monkeypatch.setattr(ToyOracle, "load", forbidden)
    data = CampaignDataset(campaign_dir)
    episode = data.sample_episode(np.random.default_rng(20), support_size=4)
    assert episode.context.shape == (4,)
    assert episode.campaign_id not in str(episode.context)


def test_collation_maps_queries_to_padded_support_without_outcome_leakage(campaign_dir):
    data = CampaignDataset(campaign_dir)
    episodes = [data.sample_episode(np.random.default_rng(seed), support_size=size)
                for seed, size in ((10, 4), (11, 8))]
    features = torch.arange(len(data) * 97, dtype=torch.float32).reshape(len(data), 97)
    batch = collate_episodes(episodes, pooled_features=features)
    assert batch.support.features.shape == (2, 8, 97)
    assert batch.support.mask.sum(axis=1).tolist() == [4, 8]
    assert batch.episode_index.tolist() == [0] * 4 + [1] * 4
    assert len(batch.query.smiles) == len(batch.lead.smiles) == 8
    for index, episode in enumerate(episodes):
        observed = batch.support.features[index, batch.support.mask[index]]
        torch.testing.assert_close(observed, features[episode.support_indices])
        torch.testing.assert_close(batch.support.values[index, :len(episode.support_indices)],
                                   torch.from_numpy(episode.support_values))
    torch.testing.assert_close(batch.delta, torch.from_numpy(np.concatenate([e.deltas for e in episodes])))
    assert not hasattr(batch.support, "goals") and not hasattr(batch.support, "campaign_id")
    assert not hasattr(batch.support, "delta")
