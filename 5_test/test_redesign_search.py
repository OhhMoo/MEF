"""Budgeted predictor-guided search: frozen state, invalids, and oracle ordering."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from emf.chem import decode_graph, smiles_to_graph
from emf.contracts import ModelConfig
from emf.redesign_contracts import FlowCondition
from emf.redesign_evaluation import evaluate_committed_proposals, load_committed_proposals
from emf.redesign_sampling import make_sampling_plan


PATH = Path(__file__).resolve().parents[1] / "4_experimental_result/control/redesign_search.py"
SPEC = importlib.util.spec_from_file_location("emf_predictor_search_control", PATH)
CONTROL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROL
SPEC.loader.exec_module(CONTROL)


class FixedResponseScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.unused_dropout = nn.Dropout(0.8)
        self.config = ModelConfig(node_dim=2, pair_dim=1, hidden_dim=8, depth=1, num_heads=2)
        self.feature_calls = 0
        self.response_calls = 0
        self.states = []

    def graph_features(self, batch):
        assert not self.training and not self.weight.requires_grad
        self.feature_calls += 1
        atoms = batch.atoms
        return torch.stack([((atoms == index) & batch.masks.nodes).sum(axis=1)
                            for index in range(4)], dim=1).float()

    def predict_response_features(self, lead, candidate, baseline, context, state):
        assert not self.training and not self.weight.requires_grad
        self.response_calls += 1
        self.states.append(state.clone())
        mean = torch.stack((0.1 * candidate[:, 1] - 0.1 * candidate[:, 3],
                            0.1 * candidate[:, 2]), dim=1)
        covariance = torch.eye(2).expand(len(candidate), 2, 2)
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)

    def predict_clean(self, *args, **kwargs):
        pytest.fail("predictor search must not call a flow")

    def decode(self, *args, **kwargs):
        pytest.fail("categorical mutation search must not call the neural decoder")


def condition():
    return FlowCondition(torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1, 4),
                         torch.full((1, 8), 7.0), torch.tensor([[0.25, -0.1]]))


def plan_for(lead, attempts=1000):
    config = FixedResponseScorer().config
    count = smiles_to_graph(lead)["count"]
    return make_sampling_plan(config, np.eye(9), count, attempts, count_seed=70, noise_seed=71)


def test_mutation_catalogue_includes_unsanitized_and_disconnected_attempts():
    graph = smiles_to_graph("CCF")
    mutations = CONTROL.enumerate_mutations(graph)
    outcomes = [decode_graph(edit.atoms, edit.bonds, edit.count, 9) for edit in mutations]
    assert any(not outcome.chemically_valid for outcome in outcomes)
    assert any(outcome.chemically_valid and not outcome.connected for outcome in outcomes)
    assert {edit.operation for edit in mutations} == {
        "atom_substitution", "bond_change", "terminal_atom_addition", "terminal_atom_deletion"}
    assert all(1 <= edit.count <= 9 for edit in mutations)


def test_search_keeps_exact_budget_and_native_state_without_flow_or_oracle(tmp_path):
    torch.set_num_threads(1)
    lead = "CCOC(=O)NCC"
    model = FixedResponseScorer()
    model.weight.grad = torch.tensor(3.0)
    model.unused_dropout.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rows, ranking, summary = CONTROL.generate_predictor_search_proposals(
        model, condition(), plan_for(lead), lead, [lead, "CCNC(=O)OCC"], tmp_path,
        batch_size=32)
    assert len(rows) == summary["attempts"] == 1000
    assert summary["nfe"] == summary["actual_flow_evaluations"] == summary["actual_decode_evaluations"] == 0
    assert summary["actual_response_scoring_calls"] == model.response_calls > 0
    assert summary["actual_graph_feature_calls"] == model.feature_calls
    assert summary["actual_model_calls"] == model.feature_calls + model.response_calls
    assert summary["invalid_attempts"] > 0 and summary["disconnected_attempts"] > 0
    assert summary["duplicate_attempts"] > 0
    assert all(state.eq(7).all() for state in model.states)
    assert all(row["nfe"] == 0 for row in rows)
    assert [module.training for module in model.modules()] == modes
    assert model.weight.requires_grad and model.weight.grad.item() == 3
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert len(ranking) <= 16
    assert not (tmp_path / "evaluation_outcomes.jsonl").exists()
    committed_rows, committed_ranking, committed = load_committed_proposals(tmp_path)
    assert len(committed_rows) == 1000 and committed_ranking == ranking
    assert not committed["oracle_access_before_commit"]
    calls = []

    def evaluator(smiles):
        load_committed_proposals(tmp_path)
        calls.append(smiles)
        return [1.0, 0.0]

    observed = evaluate_committed_proposals(tmp_path, evaluator, baseline=[0, 0],
                                             goal=[0.25, -0.1], support_size=16)
    assert observed["attempts"] == 1000
    assert len(calls) == len(set(calls))
    assert all(smiles not in {lead, "CCNC(=O)OCC"} for smiles in calls)


def test_search_is_deterministic_at_fixed_mutation_seed_and_batch_size(tmp_path):
    lead = "CCOC(=O)NCC"
    plan = plan_for(lead, 256)
    left = CONTROL.generate_predictor_search_proposals(
        FixedResponseScorer(), condition(), plan, lead, [lead], tmp_path / "left",
        num_attempts=256, batch_size=32)
    right = CONTROL.generate_predictor_search_proposals(
        FixedResponseScorer(), condition(), plan, lead, [lead], tmp_path / "right",
        num_attempts=256, batch_size=32)
    comparable = lambda rows: [(row["parent_smiles"], row["mutation_operation"], row["atoms"],
                                row["bonds"], row["profile_score"], row["selected_rank"]) for row in rows]
    assert comparable(left[0]) == comparable(right[0])
    assert left[1] == right[1]


def test_attempt_mismatch_fails_before_scoring(tmp_path):
    model = FixedResponseScorer()
    with pytest.raises(ValueError, match="same attempted budget"):
        CONTROL.generate_predictor_search_proposals(model, condition(), plan_for("CC", 20),
                                                     "CC", ["CC"], tmp_path, num_attempts=10)
    assert model.feature_calls == model.response_calls == 0


def test_control_uses_real_source_model_graph_features_and_joint_response(tmp_path):
    from emf.data import collate_graphs
    from emf.redesign import RedesignModel
    from emf.redesign_contracts import RedesignConfig

    torch.set_num_threads(1)
    torch.manual_seed(22)
    graph_config = ModelConfig(node_dim=2, pair_dim=1, hidden_dim=8, depth=1, num_heads=2)
    model = RedesignModel(RedesignConfig(model=graph_config, state_dim=8))
    lead = "CCOC(=O)NCC"
    lead_batch = collate_graphs([{**smiles_to_graph(lead), "id": "lead"}])
    native = model.condition(lead_batch, torch.zeros(1, 2), torch.zeros(1, 4),
                             torch.ones(1, 8), torch.tensor([[0.25, -0.1]]))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rows, ranking, summary = CONTROL.generate_predictor_search_proposals(
        model, native, plan_for(lead, 64), lead, [lead], tmp_path,
        num_attempts=64, batch_size=16)
    assert len(rows) == 64 and summary["actual_response_scoring_calls"] > 0
    assert any(row["profile_score"] is not None for row in rows)
    assert not any(row["scoring_failure"] for row in rows)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
