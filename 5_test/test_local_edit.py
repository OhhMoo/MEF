"""Meaningful alignment and slot-denominator checks for the local-edit study."""

from itertools import permutations

import numpy as np
import torch

from emf.chem import decode_graph, smiles_to_graph
from emf.contracts import GraphMasks, Latents
from emf.local_edit_control import (_minimum_norm_two_constraints, _source_training_rows,
                                    calibrate_coefficient, correct_clean)
from emf.local_edit_data import AlignedSteeringDataset, aligned_graph, align_family_dataset
from emf.local_edit_evaluation import audit_attempts
from emf.steering_data import REFERENCE_SMILES, SteeringDataset, SteeringOracle


def _matrix(graph):
    n = graph["count"]
    matrix = np.zeros((n, n), dtype=np.uint8)
    upper = np.triu_indices(n, k=1)
    matrix[upper] = graph["bonds"]
    return matrix + matrix.T


def test_alignment_is_global_minimum_with_deterministic_tie():
    lead, candidate = smiles_to_graph("CCNO", 4), smiles_to_graph("CNCO", 4)
    aligned = aligned_graph(lead["smiles"], candidate["smiles"], 4)
    a, b = _matrix(lead), _matrix(candidate)
    costs = []
    for mapping in permutations(range(4)):
        cost = sum(candidate["atoms"][i] != lead["atoms"][mapping[i]] for i in range(4))
        cost += sum(b[i, j] != a[mapping[i], mapping[j]]
                    for i in range(4) for j in range(i + 1, 4))
        costs.append((int(cost), mapping))
    assert (aligned["total_edits"], tuple(aligned["canonical_to_lead"])) == min(costs)
    assert aligned["atom_edits"] + aligned["bond_edits"] == aligned["total_edits"]
    assert decode_graph(aligned["atoms"], aligned["bonds"], 4, 4).smiles == candidate["smiles"]


def test_declared_family_cache_reconstructs_every_molecule_and_lead(tmp_path):
    dataset = SteeringDataset("2_data_storage/processed/evidence_steering_toy", "train")
    output = tmp_path / "alignment.json"
    cache = align_family_dataset(dataset, output)
    assert output.exists()
    assert sum(map(len, cache.values())) == sum(map(len, dataset.family_indices))
    for family, lead_index in enumerate(dataset.family_leads):
        lead = cache[family][lead_index]
        assert lead["total_edits"] == 0
        assert lead["canonical_to_lead"].tolist() == list(range(9))
        for index, graph in cache[family].items():
            assert decode_graph(graph["atoms"], graph["bonds"], 9, 9).smiles == dataset.smiles[index]
            assert sorted(graph["canonical_to_lead"].tolist()) == list(range(9))
            assert graph["atom_edits"] + graph["bond_edits"] == graph["total_edits"]

    for split in ("train", "validation", "test"):
        aligned_dataset = AlignedSteeringDataset(
            "2_data_storage/processed/evidence_steering_toy", split, alignment_path=output)
        episode = aligned_dataset.sample_episode(np.random.default_rng(12), support_size=4,
                                                 queries_per_lead=8, campaign_index=0)
        assert episode.lead_graph["smiles"] == aligned_dataset.smiles[episode.lead_index]
        for index, graph in zip(episode.query_indices, episode.query_graphs):
            assert graph["id"] == aligned_dataset.ids[index]
            assert graph["smiles"] == aligned_dataset.smiles[index]
            assert decode_graph(graph["atoms"], graph["bonds"], 9, 9).smiles == graph["smiles"]
            assert graph["atom_edits"] + graph["bond_edits"] == graph["total_edits"]


def test_audit_keeps_invalid_duplicates_and_reserved_slots_in_denominator():
    lead = REFERENCE_SMILES[0]
    candidate = "C#CNC(=O)CC(C)C"
    good = smiles_to_graph(candidate, 9)
    source = smiles_to_graph(lead, 9)
    bad = dict(good, atoms=np.full(9, 9, dtype=np.uint8), smiles=None)
    attempts = [good, good, source, bad]
    calls = []

    def response(smiles):
        calls.append(smiles)
        return np.asarray([0.5, 0.0])

    rows, summary = audit_attempts(lead, attempts, response_lookup=response)
    assert summary["attempts"] == 4
    assert summary["distinct_local_profile_success_attempts"] == 1
    assert summary["distinct_local_profile_yield_per_attempt"] == 0.25
    assert calls == [good["smiles"]]
    assert rows[0]["total_edits"] >= 1 and rows[0]["profile_success"]
    assert rows[1]["duplicate"] and not rows[1]["eligible"]
    assert rows[2]["reserved_support_match"] and not rows[2]["eligible"]
    assert not rows[3]["chemically_valid"] and rows[3]["total_edits"] is None


def test_audit_matches_declared_locality_and_retains_disconnected_identity():
    from emf.campaigns import neighbourhood

    lead = REFERENCE_SMILES[0]
    nearby = smiles_to_graph("CCNC(=O)CC(C)", 9)
    disconnected = {"atoms": np.zeros(9, dtype=np.uint8),
                    "bonds": np.zeros(36, dtype=np.uint8), "count": 9}
    disconnected["smiles"] = decode_graph(
        disconnected["atoms"], disconnected["bonds"], 9, 9).smiles
    rows, summary = audit_attempts(lead, [nearby, disconnected])
    assert rows[0]["within_local_edit_rule"] == neighbourhood(nearby["smiles"], lead)["eligible"]
    assert rows[0]["total_edits"] is None  # count changed; primary locality still evaluated
    assert rows[1]["smiles"] == disconnected["smiles"]
    assert rows[1]["chemically_valid"] and not rows[1]["connected"]
    assert not rows[1]["eligible"] and summary["attempts"] == 2
    bounded, _ = audit_attempts(lead, [nearby], max_edits=0)
    assert not bounded[0]["within_local_edit_rule"]


def test_source_surrogate_targets_use_only_source_coefficients():
    path = "2_data_storage/processed/evidence_steering_toy"
    train = SteeringDataset(path, "train")
    oracle = SteeringOracle.load(f"{path}/private/oracle.json")
    leads, queries, targets = _source_training_rows(train, oracle)
    assert len(targets) == sum(len(train.family_indices[train.campaign_family[name]])
                               for name in train.campaign_ids)
    np.testing.assert_allclose(targets[queries == leads], 0, atol=1e-7)
    assert np.isfinite(targets).all()
    validation = SteeringDataset(path, "validation")
    try:
        _source_training_rows(validation, oracle)
    except ValueError:
        pass
    else:
        raise AssertionError("held-out observations entered surrogate fit")


def test_two_endpoint_projection_preserves_already_satisfied_endpoint():
    jacobian = torch.tensor([[[1., 0.], [-1., 1.]],
                             [[1., 0.], [-1., 0.]]])
    deficit = torch.tensor([[1., -0.2], [1., 0.]])
    delta, feasible = _minimum_norm_two_constraints(jacobian, deficit, 1e-4)
    torch.testing.assert_close(delta[0], torch.tensor([1., 0.8]), atol=1e-5, rtol=0)
    assert feasible.tolist() == [True, False]
    assert torch.all((jacobian[0] @ delta[0]) >= deficit[0] - 1e-6)


def test_small_well_conditioned_jacobian_remains_exact():
    jacobian = torch.eye(2)[None] * 1e-6
    delta, feasible = _minimum_norm_two_constraints(
        jacobian, torch.tensor([[1e-3, 2e-3]]), 1e-4)
    assert feasible.tolist() == [True]
    torch.testing.assert_close(delta[0], torch.tensor([1000., 2000.]), atol=0.1, rtol=1e-4)


def test_coefficient_calibration_uses_measured_support_only():
    class UnitResponse(torch.nn.Module):
        def forward(self, lead, candidate):
            return torch.stack((candidate[:, 0] - lead[:, 0],
                                torch.zeros_like(candidate[:, 0])), dim=-1)

    lead = torch.zeros(1, 2)
    features = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [-1., 0.]])
    measurements = torch.tensor([[0., 0.], [2., 3.], [4., -2.], [-2., 1.]])
    coefficient, report = calibrate_coefficient(UnitResponse(), lead, features, measurements)
    torch.testing.assert_close(coefficient, torch.tensor(2.), atol=1e-6, rtol=0)
    assert report["uses_support_measurements_only"] and report["informative_support"]


def test_clean_correction_masks_inactive_coordinates_and_counts_calls():
    class LinearSurrogate(torch.nn.Module):
        def forward(self, lead_features, candidate_features):
            return candidate_features[:, :2] - lead_features[:, :2]

    masks = GraphMasks.from_counts(torch.tensor([1]), 2)
    clean = Latents(torch.zeros(1, 2, 2), torch.zeros(1, 1, 1))
    lead_features = torch.zeros(1, 4)
    corrected, stats = correct_clean(
        clean, masks, lead_features, torch.zeros(1, 2), torch.tensor([[1., 0.]]),
        1., LinearSurrogate(), radius=100.)
    torch.testing.assert_close(corrected.nodes[0, 0], torch.tensor([1., 0.]))
    assert corrected.nodes[0, 1].abs().sum() == 0
    assert corrected.pairs.abs().sum() == 0
    assert stats["surrogate_forward_calls"] == 2
    assert stats["surrogate_backward_calls"] == 2
    assert stats["linearized_feasible"] == [True]
