"""Independent scientific-contract checks for the B revision benchmark."""

from collections import defaultdict
import csv
from types import SimpleNamespace

import pytest
import torch

import emf.masked_edit_revision_benchmark as benchmark


def test_goal_reversal_preserves_physical_coefficients_and_endpoint_two():
    coefficients = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
    original = coefficients.clone()
    oriented = benchmark._orient(coefficients, (-1, 1))
    torch.testing.assert_close(coefficients, original)
    torch.testing.assert_close(oriented[:, 0], -original[:, 0])
    torch.testing.assert_close(oriented[:, 1], original[:, 1])


@pytest.mark.parametrize("physical_delta, expected", [
    ((-.4, -.05), True), ((-.4, -.2), False), ((.4, -.05), False)])
def test_reversal_scores_new_goal_and_retained_endpoint_after_commit(
        monkeypatch, physical_delta, expected):
    calls = []
    predicted = torch.tensor([[-physical_delta[0], physical_delta[1]]])
    monkeypatch.setattr(benchmark, "score_prediction", lambda *args: (None, predicted))
    monkeypatch.setattr(benchmark, "graphs_from_logits", lambda logits: [{"committed": True}])

    class Evaluator:
        def evaluate_graph(self, graph, mask, cid, **kwargs):
            calls.append((graph, kwargs))
            return dict(exact_delta=list(physical_delta), chemically_valid=True,
                        connected=True, protected_copy=True, lead_copy=False,
                        support_match=False, source_match=False,
                        original_locality=True, smiles="candidate", goal_success=False)

    dataset = SimpleNamespace(masks={"mask": object()},
                              manifest={"campaigns": {"case": {"mask_id": "mask"}}},
                              smiles=["source"], source_graph_indices=[0])
    reserve = ["same-support"]
    rows = benchmark._score_rows(None, None, None, None, None, None, None,
                                 dataset, "case", {"reserve_smiles": reserve}, Evaluator(),
                                 scenario="goal_reversal", method="continue_control",
                                 target=.5, first=0, sign=(-1, 1), seed=142,
                                 split="validation", cost={})
    assert calls[0][0]["committed"]
    assert calls[0][1]["support_smiles"] is reserve
    assert rows[0]["physical_exact_delta"] == list(physical_delta)
    assert rows[0]["goal_success"] is expected
    assert rows[0]["eligible_novel_success"] is expected
    assert rows[0]["endpoint2_preserved"] is (physical_delta[1] >= -.1)


def _attempt(sample, seconds):
    return dict(sample=sample, charged_seconds_per_attempt=seconds,
                batch_first=sample // 2 * 2, old_goal_success=False,
                eligible=True, rank_score=1., smiles=f"candidate-{sample}",
                chemically_valid=True, connected=True, goal_success=True,
                endpoint2_preserved=True)


def test_budget_frontier_keeps_whole_batches_and_empty_slots():
    ledger = defaultdict(list)
    key = ("campaign", "goal_reversal", "conditional_continue", .5)
    ledger[key] = [_attempt(i, 1.) for i in range(4)]
    rows = benchmark._budget_rows(ledger, (1.5, 2., 3.9, 4.), 16)
    assert [row["attempts"] for row in rows] == [0, 2, 2, 4]
    assert [row["topk_successes"] for row in rows] == [0, 2, 2, 4]
    assert [row["topk_success_fraction"] for row in rows] == [0., .125, .125, .25]
    assert [row["charged_seconds_used"] for row in rows] == [0., 2., 2., 4.]


def test_integrated_conditional_prefix_and_shared_support(tmp_path, monkeypatch):
    from emf.masked_edit_control import SoftGraphDescriptorCritic
    from emf.masked_edit_run import new_model

    torch.set_num_threads(1)
    model = new_model("B", "cpu").eval().requires_grad_(False)
    conditional = new_model("B", "cpu", conditioned=True).eval().requires_grad_(False)
    critic = SoftGraphDescriptorCritic(model.config, hidden_dim=16, depth=1).eval().requires_grad_(False)
    monkeypatch.setattr(benchmark, "load_model", lambda path, device:
                        (conditional if path == "conditional" else model,
                         {"variant": "B", "conditioned": path == "conditional"}))
    monkeypatch.setattr(benchmark, "load_critic", lambda *args: (critic, {}))
    original_score = benchmark._score_rows
    supports = []

    def capture_support(*args, **kwargs):
        supports.append(tuple(sorted(args[9]["reserve_smiles"])))
        return original_score(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_score_rows", capture_support)
    benchmark.run(dict(checkpoint="prior", conditional_checkpoint="conditional",
                       critic="critic", output=str(tmp_path), seed=142,
                       split="validation", campaign_limit=1, attempts=2,
                       baseline_attempts=2, chunk=2, nfe=3, iterations=0,
                       revision_times=[.5], wall_budgets_seconds=[100.]), "cpu")
    assert supports and len(set(supports)) == 1
    with (tmp_path / "attempts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    continuing = [row for row in rows if row["method"] == "conditional_continue"]
    restarting = [row for row in rows if row["method"] == "conditional_restart"]
    controlled = [row for row in rows if row["method"] == "restart_control"]
    assert continuing and restarting and controlled
    assert {int(row["charged_forward_nfe"]) for row in continuing} == {3}
    assert {int(row["charged_forward_nfe"]) for row in restarting} == {4}
    assert {int(row["charged_forward_nfe"]) for row in controlled} == {7}
    for row in continuing + restarting:
        assert int(row["old_forward_nfe"]) == int(row["charged_backward_nfe"]) == 0
        assert int(row["prefix_forward_nfe"]) == 1
