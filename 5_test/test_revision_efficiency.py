"""Operational checks for the paired B revision efficiency runner."""

import csv
from collections import defaultdict
from unittest.mock import patch

import pytest

from emf.masked_edit_control import SoftGraphDescriptorCritic
from emf.masked_edit_revision_benchmark import _budget_rows, run
from emf.masked_edit_run import new_model


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_test_split_needs_validation_selected_budgets(tmp_path):
    with pytest.raises(ValueError, match="validation-selected"):
        run({"checkpoint": "unused", "critic": "unused", "output": str(tmp_path),
             "seed": 142, "split": "test", "attempts": 2,
             "baseline_attempts": 2, "nfe": 3}, "cpu")


def test_small_paired_operation_reconstructs_batch_costs_and_retains_mutations(tmp_path):
    model = new_model("B", "cpu").eval().requires_grad_(False)
    critic = SoftGraphDescriptorCritic(model.config, hidden_dim=16,
                                       depth=1).eval().requires_grad_(False)
    with patch("emf.masked_edit_revision_benchmark.load_model",
               return_value=(model, {"variant": "B"})), patch(
               "emf.masked_edit_revision_benchmark.load_critic",
               return_value=(critic, {})):
        result = run({"checkpoint": "mock", "critic": "mock",
                      "output": str(tmp_path), "seed": 142,
                      "split": "validation", "campaign_limit": 1,
                      "attempts": 2, "baseline_attempts": 4,
                      "chunk": 2, "nfe": 3, "iterations": 0,
                      "revision_times": (.5,),
                      "wall_budgets_seconds": [.000001, 1000.]}, "cpu")
    attempts = _csv(tmp_path / "attempts.csv")
    cases = _csv(tmp_path / "cases.csv")
    batches = _csv(tmp_path / "cost_batches.csv")
    frontier = _csv(tmp_path / "budget_frontier.csv")
    feasibility = _csv(tmp_path / "feasibility.csv")
    assert result["logical_attempt_rows"] == len(attempts)
    assert {"goal_reversal", "evidence_revision"} == {row["scenario"] for row in cases}
    assert "direct_mutation" in {row["method"] for row in cases}
    assert len(feasibility) == 1
    assert int(feasibility[0]["reference_attempts"]) >= int(feasibility[0]["eligible_distinct"])
    assert {"total", "incremental"} == {row["budget_basis"] for row in frontier}
    assert all(row["support_match"] == "False" for row in attempts if row["eligible"] == "True")
    assert all(int(row["attempts"]) == 0 and int(row["topk_successes"]) == 0
               for row in frontier if float(row["wall_budget_seconds"]) < .001)
    for case in cases:
        matching = [row for row in batches if
                    all(row[field] == case[field] for field in
                        ("campaign_id", "scenario", "method", "revision_target"))]
        assert sum(int(row["batch_count"]) for row in matching) == int(case["attempts"])
        assert sum(int(row["charged_forward_nfe"]) for row in matching) == int(case["charged_forward_nfe"])
        assert sum(int(row["charged_backward_nfe"]) for row in matching) == int(case["charged_backward_nfe"])


def test_incremental_budget_selects_complete_batches_without_discarded_history():
    ledger = defaultdict(list)
    key = ("campaign", "evidence_revision", "continue_control", .5)
    for first in (0, 2):
        for sample in (first, first+1):
            ledger[key].append(dict(sample=sample, batch_first=first,
                charged_seconds_per_attempt=5., incremental_seconds_per_attempt=1.,
                eligible=True, rank_score=float(sample), smiles=str(sample),
                chemically_valid=True, connected=True, goal_success=True,
                endpoint2_preserved=True, old_goal_success=True))
    total = _budget_rows(ledger, (4.,), 16, basis="total")
    incremental = _budget_rows(ledger, (4.,), 16, basis="incremental")
    assert total[0]["attempts"] == 0
    assert incremental[0]["attempts"] == 4
    assert incremental[0]["total_charged_seconds"] == 20.
    assert incremental[0]["incremental_charged_seconds"] == 4.
