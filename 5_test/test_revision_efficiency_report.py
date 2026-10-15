"""Small reporting-contract tests using primary-artifact fixtures only."""

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "3_model_training" / "scripts"))
import revision_efficiency_report as report


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_fixture(study: Path, seed: int, hits_by_campaign):
    run = study / "test" / f"seed{seed}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "phase": "complete", "stage": "revision_efficiency", "seed": seed,
        "split": "test", "variant": "B", "campaigns": 3,
        "completed_utc": f"2026-10-14T16:{seed % 60:02d}:00+00:00",
        "wall_budgets_seconds": [5.0], "physical_exact_response_calls": 9,
    }), encoding="utf-8")
    case_rows, frontier_rows, batch_rows = [], [], []
    for campaign_index, campaign in enumerate(("c0", "c1", "c2")):
        for method in ("continue_control", "restart_control"):
            hits = hits_by_campaign[method][campaign_index]
            case_rows.append({
                "campaign_id": campaign, "scenario": "evidence_refinement",
                "method": method, "revision_target": .5, "seed": seed, "split": "test",
                "attempts": 16, "valid_connected_attempts": 12,
                "eligible_attempts": 10, "distinct_eligible": 10,
                "distinct_goal_successes": hits, "success_attempts": hits,
                "endpoint2_preserved_attempts": 10, "topk": 16,
                "topk_filled": 12, "topk_successes": hits,
                "topk_success_fraction": hits / 16,
                "total_charged_seconds": 10.0 if method == "continue_control" else 12.0,
                "incremental_charged_seconds": 4.0 if method == "continue_control" else 6.0,
                "charged_forward_nfe": 100, "charged_backward_nfe": 20,
                "incremental_forward_nfe": 30, "incremental_backward_nfe": 5,
            })
            frontier_rows.append({
                "campaign_id": campaign, "scenario": "evidence_refinement",
                "method": method, "revision_target": .5,
                "budget_basis": "total",
                "wall_budget_seconds": 5.0, "charged_seconds_used": 5.0,
                "unused_budget_seconds": 0.0,
                "cap_limited": "true" if campaign == "c2" else "false",
                "attempts": 8, "valid_connected_attempts": 6,
                "eligible_attempts": 5, "distinct_eligible": 5,
                "distinct_goal_successes": hits // 2, "success_attempts": hits // 2,
                "endpoint2_preserved_attempts": 5, "topk": 16,
                "topk_filled": 8, "topk_successes": hits // 2,
                "topk_success_fraction": (hits // 2) / 16,
                "total_charged_seconds": 5.0,
                "incremental_charged_seconds": 2.0,
            })
            frontier_rows.append({**frontier_rows[-1], "budget_basis": "incremental",
                                  "wall_budget_seconds": 2.0})
            batch_rows.append({
                "campaign_id": campaign, "scenario": "evidence_refinement",
                "method": method, "revision_target": .5, "batch_first": 0,
                "batch_count": 8, "total_charged_seconds": 5.0,
                "incremental_charged_seconds": 2.0,
                "cumulative_charged_seconds": 5.0,
                "cumulative_incremental_seconds": 2.0,
                "charged_forward_nfe": 50,
                "cumulative_charged_forward_nfe": 50,
                "charged_backward_nfe": 10,
                "cumulative_charged_backward_nfe": 10,
            })
    _write_csv(run / "cases.csv", sorted(case_rows[0]), case_rows)
    _write_csv(run / "budget_frontier.csv", sorted(frontier_rows[0]), frontier_rows)
    _write_csv(run / "cost_batches.csv", sorted(batch_rows[0]), batch_rows)


def test_refresh_ignores_derived_and_snapshot_summaries(tmp_path):
    study = tmp_path / "study"
    # These are deliberate traps: their contents are malformed and must never be read.
    for directory in (study / "analysis" / "trap", study / "source_snapshot_fake" / "trap"):
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("not json", encoding="utf-8")

    first = report.build_report(study)
    second = report.build_report(study)
    assert first["status"] == second["status"] == "pending"
    assert first["completed_runs"] == second["completed_runs"] == []
    assert first["case_summaries_by_seed"] == second["case_summaries_by_seed"] == []
    assert (study / "analysis" / "report.json").is_file()
    assert (study / "live_report.md").is_file()


def test_paired_bootstrap_keeps_all_seeds_inside_campaign_clusters(tmp_path):
    study = tmp_path / "study"
    # Frontier paired effects by campaign are 1/16, 0/16, and 1/16; each repeats
    # across three seeds, which must remain one cluster per campaign.
    for seed in report.SEED_SET:
        _run_fixture(study, seed,
                     {"continue_control": [4, 8, 12],
                      "restart_control": [2, 8, 10]})
    retry = study / "test" / "seed142_retry1"
    shutil.copytree(study / "test" / "seed142", retry)
    retry_summary = json.loads((retry / "summary.json").read_text(encoding="utf-8"))
    retry_summary["completed_utc"] = "2026-10-15T00:00:00+00:00"
    (retry / "summary.json").write_text(json.dumps(retry_summary), encoding="utf-8")
    operation = study / "operation" / "seed142"
    shutil.copytree(retry, operation)
    operation_summary = json.loads((operation / "summary.json").read_text(encoding="utf-8"))
    operation_summary["completed_utc"] = "2030-01-01T00:00:00+00:00"
    (operation / "summary.json").write_text(json.dumps(operation_summary), encoding="utf-8")

    result = report.build_report(study)
    assert len(result["completed_runs"]) == 3
    assert all("operation" not in row["path"] for row in result["completed_runs"])
    assert any(row["path"] == "test/seed142_retry1"
               for row in result["completed_runs"])
    assert result["superseded_run_directories"] == [{
        "path": "test/seed142", "split": "test", "seed": 142,
        "campaigns_replaced": 3, "selected_run_paths": ["test/seed142_retry1"]}]
    assert len(result["case_summaries_by_seed"]) == 6
    frontier = result["paired_campaign_bootstrap"]
    assert len(frontier) == 2
    assert {row["budget_basis"] for row in frontier} == {"total", "incremental"}
    for contrast in frontier:
        assert contrast["paired_campaigns"] == 3
        assert contrast["seeds"] == [142, 143, 144]
        assert contrast["bootstrap_samples"] == 2000
        assert contrast["mean_continue_minus_restart"] == pytest.approx(1 / 24)
        assert contrast["status"] == "complete"
        assert contrast["ci95_low"] <= contrast["mean_continue_minus_restart"]
        assert contrast["mean_continue_minus_restart"] <= contrast["ci95_high"]
    frontier_summary = result["budget_summaries_by_seed"]
    assert len(frontier_summary) == 12
    assert all(row["unused_slots"] == 24 for row in frontier_summary)
    assert all(row["cap_limited_cases"] == 1 for row in frontier_summary)


def test_missing_static_b_and_empty_study_are_reported_as_pending(tmp_path):
    result = report.build_report(tmp_path / "empty")
    assert result["status"] == "pending"
    assert result["missing_static_b_inputs"] == ["static_b/seed*/metrics.csv"]
    assert "Static B `run_control` metrics are pending." in (
        tmp_path / "empty" / "live_report.md").read_text(encoding="utf-8")
