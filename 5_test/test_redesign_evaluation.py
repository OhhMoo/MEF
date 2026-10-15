"""Check Gaussian event scoring, budgeted ledgers, and oracle access order."""

import hashlib
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from emf.contracts import GraphLogits, Latents, ModelConfig
from emf.redesign_contracts import FlowCondition
from emf.redesign_evaluation import (commit_proposals, evaluate_committed_proposals,
    generate_redesign_proposals, joint_profile_probability, top_k_candidates)
from emf.redesign_sampling import make_sampling_plan


@pytest.mark.parametrize("rho", [-0.999999999, -0.8, 0, 0.8, 0.999999999])
def test_joint_probability_uses_full_covariance(rho):
    covariance = [[1, rho], [rho, 1]]
    actual = joint_profile_probability([0, 0], covariance, [0, 0], [0, 0])
    expected = 0.25 + math.asin(rho) / (2 * math.pi)
    assert actual == pytest.approx(expected, abs=2e-8)


def test_probability_broadcasts_and_moves_the_final_endpoint_threshold():
    result = joint_profile_probability([[0, 0], [0, 0]], np.eye(2),
                                        [[0, 0], [0, 1]], [0, 0])
    assert result.tolist() == pytest.approx([0.25, 0.5 * 0.8413447460685429])
    with pytest.raises(ValueError, match="positive definite"):
        joint_profile_probability([0, 0], [[1, 2], [2, 1]], [0, 0], [0, 0])


class EthaneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.config = ModelConfig(max_nodes=2, node_dim=2, pair_dim=1,
                                  hidden_dim=8, depth=1, num_heads=2)
        self.flow_calls = []
        self.decode_calls = 0
        self.scorer_states = []

    def predict_clean(self, z, t, masks, condition):
        assert not self.training and not self.weight.requires_grad
        self.flow_calls.append(condition.response_state.clone())
        return Latents(torch.zeros_like(z.nodes), torch.zeros_like(z.pairs))

    def decode(self, latents, masks):
        self.decode_calls += 1
        atoms = torch.zeros(*masks.nodes.shape, 4)
        bonds = torch.zeros(*masks.pairs.shape, 4)
        atoms[..., 0] = 1
        bonds[..., 1] = 1
        return GraphLogits(atoms, bonds)

    def graph_features(self, graphs):
        return torch.ones(len(graphs.atoms), 4)

    def predict_response_features(self, lead, candidate, baseline, context, state):
        self.scorer_states.append(state.clone())
        return torch.distributions.MultivariateNormal(torch.tensor([[1.0, 0.0]]).expand(len(candidate), -1),
            covariance_matrix=torch.eye(2).expand(len(candidate), 2, 2))


def test_frozen_generation_commits_before_truth_and_retains_duplicate_attempts(tmp_path):
    model = EthaneModel()
    condition = FlowCondition(torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1, 4),
                              torch.ones(1, 8), torch.tensor([[0.5, -0.1]]))
    plan = make_sampling_plan(model.config, [[0, 1], [0, 1]], 1, 5)
    ledger, ranking, generation = generate_redesign_proposals(model, condition, plan, "C", ["C"], tmp_path,
        neighbourhood_fn=lambda *args: {"similarity": 0.5, "count_difference": 1, "eligible": True},
        batch_size=2, nfe=32, k=4, flow_state_override=torch.zeros(1, 8))
    assert len(ledger) == generation["attempts"] == 5
    assert len(model.flow_calls) == 3 * 32 and model.decode_calls == 3
    assert all(not value.any() for value in model.flow_calls)
    assert all(value.eq(1).all() for value in model.scorer_states)
    assert [row["duplicate"] for row in ledger] == [False, True, True, True, True]
    assert len(ranking) == 1 and ranking[0]["smiles"] == "CC"
    assert model.training and model.weight.requires_grad
    calls = []

    def oracle(smiles):
        assert (tmp_path / "proposal_commit.json").exists()
        committed = json.loads((tmp_path / "proposal_commit.json").read_text())
        assert committed["attempts_sha256"] == hashlib.sha256((tmp_path / "attempts.jsonl").read_bytes()).hexdigest()
        calls.append(smiles)
        return [1.0, 0.0]

    result = evaluate_committed_proposals(tmp_path, oracle, baseline=[0, 0], goal=[0.5, -0.1], support_size=16, k=4)
    assert calls == ["CC"]
    assert result["success"] == 0.25 and result["efficiency"] == 1 / 20
    assert result["unfilled_selection_slots"] == 3
    assert result["pre_ranking_unmeasured_joint_profile_yield_per_attempt"] == 1 / 5
    assert result["full_pool_outcome_reader_calls"] == 1


def test_separate_redesign_generator_uses_own_evidence_and_common_native_scorer(tmp_path):
    from dataclasses import replace

    scorer, generator = EthaneModel(), EthaneModel()
    native = FlowCondition(torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1, 4),
                           torch.ones(1, 8), torch.tensor([[0.5, -0.1]]))
    proposal = replace(native, response_state=torch.full((1, 8), 7.0))
    plan = make_sampling_plan(scorer.config, [[0, 1], [0, 1]], 1, 3)
    rows, ranking, summary = generate_redesign_proposals(scorer, native, plan, "C", ["C"], tmp_path,
        neighbourhood_fn=lambda *args: {"similarity": 0.5, "count_difference": 1, "eligible": True},
        nfe=4, batch_size=2, proposal_model=generator, proposal_condition=proposal)
    assert len(rows) == 3 and len(ranking) == 1
    assert not scorer.flow_calls and scorer.decode_calls == 0
    assert len(generator.flow_calls) == 8 and generator.decode_calls == 2
    assert all(value.eq(7).all() for value in generator.flow_calls)
    assert not generator.scorer_states
    assert all(value.eq(1).all() for value in scorer.scorer_states)
    assert summary["proposal_kind"] == "separate_redesign"
    assert (tmp_path / "proposal_commit.json").exists()
    assert not (tmp_path / "evaluation_outcomes.jsonl").exists()


def test_sanitized_charged_outputs_remain_attempts_without_scoring_or_oracle(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import emf.redesign_evaluation as evaluation

    monkeypatch.setattr(evaluation, "decode_graph", lambda *args, **kwargs:
        SimpleNamespace(smiles="C[NH3+]", chemically_valid=True, connected=True, failure_reason=None))
    model = EthaneModel()
    native = FlowCondition(torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1, 4),
                           torch.ones(1, 8), torch.tensor([[0.5, -0.1]]))
    plan = make_sampling_plan(model.config, [[0, 1], [0, 1]], 1, 3)
    rows, ranking, summary = generate_redesign_proposals(model, native, plan, "C", ["C"], tmp_path,
        neighbourhood_fn=lambda *args: pytest.fail("unsupported chemistry must not reach neighbourhood"),
        nfe=4, k=16)
    assert len(rows) == summary["attempts"] == 3 and not ranking
    assert all(row["chemically_valid"] and row["connected"] for row in rows)
    assert all(row["supported_chemistry"] is False and row["profile_score"] is None for row in rows)
    assert all(row["failure_reason"] == "unsupported_chemistry:formal_charge" for row in rows)
    assert not model.scorer_states
    truth = evaluate_committed_proposals(tmp_path, lambda smiles: pytest.fail("unsupported chemistry must not reach oracle"),
        baseline=[0, 0], goal=[0.5, -0.1], support_size=16)
    assert truth["external_outcome_reader_calls"] == 0 and truth["success"] == 0


def test_reserved_pool_and_external_failure_slots_remain_in_denominator(tmp_path):
    rows = [{"attempt_id": 0, "smiles": "CC", "chemically_valid": True, "connected": True,
             "reserved_support_match": True, "within_neighbourhood": True,
             "eligible": False, "profile_score": 0.9},
            {"attempt_id": 1, "smiles": "CN", "chemically_valid": True, "connected": True,
             "reserved_support_match": False, "within_neighbourhood": True,
             "eligible": True, "profile_score": 0.8},
            {"attempt_id": 2, "smiles": None, "chemically_valid": False, "connected": False,
             "eligible": False, "profile_score": None}]
    ranking = top_k_candidates(rows, k=2)
    commit_proposals(tmp_path, rows, ranking)
    seen = []

    def oracle(smiles):
        seen.append(smiles)
        return None

    result = evaluate_committed_proposals(tmp_path, oracle, baseline=[0, 0], goal=[1, 0], support_size=128, k=2)
    assert seen == ["CN"] and result["attempts"] == 3
    assert result["success"] == 0 and result["unfilled_selection_slots"] == 1
    assert result["actual_selected_complete_outcomes"] == 0


def test_modified_commit_is_rejected_before_oracle_access(tmp_path):
    commit_proposals(tmp_path, [], [])
    (tmp_path / "attempts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate_committed_proposals(tmp_path, lambda smiles: pytest.fail("oracle must remain hidden"),
            baseline=[0, 0], goal=[1, 0], support_size=16)


def test_retrospective_selection_requires_recorded_library_eligibility(tmp_path):
    rows = [{"attempt_id": 0, "smiles": "CN", "chemically_valid": True, "connected": True,
             "eligible": True, "profile_score": 0.8}]
    commit_proposals(tmp_path, rows, top_k_candidates(rows))
    with pytest.raises(ValueError, match="measured library"):
        evaluate_committed_proposals(tmp_path, lambda smiles: pytest.fail("oracle must remain hidden"),
            baseline=[0, 0], goal=[1, 0], support_size=16, evaluation_kind="recorded_library")


def test_real_redesign_modules_sample_score_and_keep_all_state_frozen(tmp_path):
    from emf.chem import smiles_to_graph
    from emf.data import collate_graphs
    from emf.models import EMFModel
    from emf.redesign import RedesignModel
    from emf.redesign_contracts import RedesignConfig, SupportBatch

    torch.set_num_threads(1)
    config = ModelConfig(max_nodes=2, node_dim=2, pair_dim=1,
                         hidden_dim=8, depth=1, num_heads=2)
    representation = EMFModel(config)
    representation.normalizer.fitted.fill_(True)
    with torch.no_grad():
        representation.decoder.node_head[-1].weight.zero_()
        representation.decoder.node_head[-1].bias.zero_()
        representation.decoder.node_head[-1].bias[0] = 2
        representation.decoder.pair_head[-1].weight.zero_()
        representation.decoder.pair_head[-1].bias.zero_()
        representation.decoder.pair_head[-1].bias[1] = 2
    model = RedesignModel(RedesignConfig(model=config, rank=2, state_dim=8), representation)
    lead_record = smiles_to_graph("C", max_nodes=2)
    lead_record["id"] = "unit-lead"
    lead = collate_graphs([lead_record])
    features = model.graph_features(lead)
    support = SupportBatch(features[:, None].expand(1, 16, -1), torch.zeros(1, 16, 2),
                           torch.zeros(1, 4), torch.ones(1, 16, dtype=torch.bool))
    state = model.support_state(support)
    query = model.condition(lead, torch.zeros(1, 2), torch.zeros(1, 4), state,
                            torch.tensor([[0.5, -0.1]]))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    plan = make_sampling_plan(model.config, [[0, 1], [0, 1]], 1, 3)
    rows, ranking, _ = generate_redesign_proposals(model, query, plan, "C", ["C"], tmp_path,
        neighbourhood_fn=lambda *args: {"similarity": 0.5, "count_difference": 1, "eligible": True},
        nfe=32, batch_size=2, k=16)
    assert len(rows) == 3 and len(ranking) == 1
    assert all(row["profile_score"] is not None for row in rows)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
