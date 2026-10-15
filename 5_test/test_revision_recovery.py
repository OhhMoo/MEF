"""Focused recovery invariants and a small integrated CPU operation."""

import csv
from collections import defaultdict
from unittest.mock import patch

import pytest
import torch

from emf.contracts import GraphMasks, Latents
from emf.masked_edit_control import SoftGraphDescriptorCritic
from emf.masked_edit_data import MaskedEditDataset
from emf.masked_edit_run import new_model, condition_for
from emf.masked_edit_sampling import make_paired_gaussian, sample_masked_edit_latents
from emf.revision_recovery import Batch, _prefix_trace, budget_frontier, perturb_masked_state, run


class _Condition:
    def __init__(self):
        self.lead = Latents(torch.ones(2, 3, 4), torch.ones(2, 3, 4))
        self.edit_masks = GraphMasks(torch.tensor([[True, False, False],
                                                   [True, False, False]]),
                                     torch.tensor([[False, True, False],
                                                   [False, True, False]]))
        self.masks = GraphMasks(torch.tensor([[True, True, False],
                                              [True, True, False]]),
                                torch.tensor([[True, True, False],
                                              [True, True, False]]))

    def validate(self, masks):
        assert masks is self.masks


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_perturbation_is_reproducible_editable_and_has_declared_rms():
    condition = _Condition()
    state = Latents(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))
    a = perturb_masked_state(state, condition, radius=.5, seed=12)
    b = perturb_masked_state(state, condition, radius=.5, seed=12)
    assert torch.equal(a.nodes, b.nodes)
    assert torch.equal(a.pairs, b.pairs)
    assert torch.equal(a.nodes[:, 1], condition.lead.nodes[:, 1])
    assert torch.equal(a.pairs[:, 0], condition.lead.pairs[:, 0])
    assert torch.count_nonzero(a.nodes[:, 2]) == 0
    assert torch.count_nonzero(a.pairs[:, 2]) == 0
    editable = torch.cat((a.nodes[:, 0], a.pairs[:, 1]), dim=1)
    assert torch.allclose(torch.sqrt((editable**2).mean(dim=1)),
                          torch.full((2,), .5), atol=1e-6)
    zero = perturb_masked_state(state, condition, radius=0, seed=12)
    assert torch.equal(zero.nodes[:, 0], state.nodes[:, 0])
    with pytest.raises(ValueError, match="nonnegative"):
        perturb_masked_state(state, condition, radius=-1, seed=12)


def test_budget_frontier_requires_completed_batch_and_counts_candidate_calls():
    def row(sample):
        return {"sample": sample, "eligible": True, "rank_score": float(sample),
                "smiles": str(sample), "chemically_valid": True,
                "connected": True, "goal_success": True,
                "endpoint2_preserved": True, "old_goal_success": False}
    batches = [Batch("c", "randomized_suffix", 0, 2, 8, 4., 1., 3.,
                     48, 24, 8, (row(0), row(1))),
               Batch("c", "randomized_suffix", 2, 2, 8, 4., 1., 3.,
                     48, 24, 8, (row(2), row(3)))]
    frontier = budget_frontier(batches, (3.9, 4., 8.), topk=16)
    total = [r for r in frontier if r["budget_basis"] == "total"]
    post = [r for r in frontier if r["budget_basis"] == "post_update"]
    assert [r["attempts"] for r in total] == [0, 2, 4]
    assert [r["candidate_attempts"] for r in total] == [0, 8, 16]
    assert [r["attempts"] for r in post] == [4, 4, 4]
    assert total[-1]["prefix_forward_nfe"] == 48


def test_cached_prefix_and_zero_intervention_reproduce_deterministic_suffix():
    torch.set_num_threads(1)
    model = new_model("B", "cpu", conditioned=True).eval().requires_grad_(False)
    dataset = MaskedEditDataset(split="validation", kind="vector")
    cid = dataset.campaign_ids[0]
    mask_id = dataset.manifest["campaigns"][cid]["mask_id"]
    condition = condition_for(model, dataset, mask_id, 1)
    draw = make_paired_gaussian(model.config, 1, 53)
    with torch.no_grad():
        full = sample_masked_edit_latents(model, condition, draw,
                                          nfe=8, T=.95, return_trace=True)
        trace, _ = _prefix_trace(model, condition, draw, nfe=8, T=.95,
                                 indices=(2, 4, 6), device="cpu")
        for index in (2, 4, 6):
            assert torch.allclose(trace[index].nodes, full.states[index].nodes, atol=1e-6)
            assert torch.allclose(trace[index].pairs, full.states[index].pairs, atol=1e-6)
        zero = perturb_masked_state(trace[6], condition, radius=0, seed=1)
        suffix = sample_masked_edit_latents(model, condition, nfe=2, T=.95,
            start_t=float(full.times[6]), initial_state=zero)
    assert suffix.nfe == 2
    assert torch.allclose(suffix.latents.nodes, full.latents.nodes, atol=1e-5)
    assert torch.allclose(suffix.latents.pairs, full.latents.pairs, atol=1e-5)


def test_small_recovery_run_retains_paired_roots_and_oracle_requests(tmp_path):
    torch.set_num_threads(1)
    model = new_model("B", "cpu", conditioned=True).eval().requires_grad_(False)
    critic = SoftGraphDescriptorCritic(model.config, hidden_dim=16,
                                       depth=1).eval().requires_grad_(False)
    with patch("emf.revision_recovery.load_model",
               return_value=(model, {"variant": "B", "conditioned": True})), patch(
               "emf.revision_recovery.load_critic", return_value=(critic, {})):
        result = run({"conditional_checkpoint": "mock", "critic": "mock",
                      "output": str(tmp_path), "seed": 142,
                      "split": "validation", "campaign_limit": 1,
                      "attempts": 2, "diagnostic_roots": 2,
                      "baseline_attempts": 2, "chunk": 2,
                      "nfe": 8, "intervention_trials": 2,
                      "search_trials": 1, "mutation_attempts": 4,
                      "wall_budgets_seconds": [1e-6, 1000.]},
                     "cpu")
    rows = _csv(tmp_path / "attempts.csv")
    cases = _csv(tmp_path / "cases.csv")
    batches = _csv(tmp_path / "cost_batches.csv")
    roots = _csv(tmp_path / "roots.csv")
    traces = _csv(tmp_path / "root_traces.csv")
    reference = _csv(tmp_path / "feasibility.csv")
    assert len(roots) == len(traces) == 2
    assert len(reference) == 1
    assert result["physical_exact_evaluation_requests"] == sum(
        row["physical_exact_call"] == "True" for row in rows)
    assert result["physical_exact_response_function_calls"] <= result[
        "physical_exact_evaluation_requests"]
    expected_methods = {"conditional_continue", "conditional_restart",
                        "rollback_25", "rollback_50", "randomized_suffix",
                        "oracle_select_same_suffix_pool", "oracle_adaptive_suffix",
                        "oracle_independent_restart", "conditional_sampling_baseline",
                        "direct_mutation", "oracle_direct_mutation"}
    assert expected_methods == {row["method"] for row in cases}
    assert {row["root_id"] for row in roots} == {row["root_draw_id"] for row in traces}
    per_case = defaultdict(list)
    for row in batches:
        per_case[row["method"]].append(row)
    reveal_nfe = int(traces[0]["reveal_grid_index"])
    for method in ("conditional_continue", "rollback_25", "rollback_50",
                   "conditional_restart", "randomized_suffix"):
        assert all(int(b["prefix_nfe"]) == reveal_nfe * int(b["count"])
                   for b in per_case[method])
        assert all(float(b["total_seconds"]) > float(b["post_seconds"])
                   for b in per_case[method])
    assert sum(int(b["exact_calls"]) for b in per_case["randomized_suffix"]) == 4
    assert sum(int(b["exact_calls"]) for b in per_case["oracle_select_same_suffix_pool"]) == 4
    assert all(row["oracle_access"] == "True" for row in rows
               if row["method"].startswith("oracle_"))
    assert all(int(row["attempts"]) == 0 for row in _csv(tmp_path / "budget_frontier.csv")
               if row["budget_basis"] == "total" and float(row["wall_budget_seconds"]) < .001)
