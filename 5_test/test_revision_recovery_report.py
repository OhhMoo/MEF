"""CSV-only reporter fixtures for retries, denominators, and paired bootstraps."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "3_model_training" / "scripts"))
import revision_recovery_report as report


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(study, stage, seed, output_name, *, delta=0):
    output = study / stage / output_name
    output.mkdir(parents=True)
    task = {"stage": stage, "seed": seed, "output": str(output), "split": stage}
    (output / "task_complete.json").write_text(json.dumps({"phase": "complete", "task": task,
        "completed_utc": f"2026-10-14T18:{seed:02d}:00+00:00"}), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"phase": "complete", "stage": stage,
        "split": stage, "seed": seed, "completed_utc": f"2026-10-14T18:{seed:02d}:00+00:00"}), encoding="utf-8")
    cases = []
    frontier = []
    for campaign in ("c0", "c1"):
        for method, success, rawvalid in (("conditional_continue", 4, 3),
                                           ("conditional_restart", 2, 4),
                                           ("rollback_25", 8 + delta, 2)):
            cases.append({"campaign_id": campaign, "method": method, "topk": 16,
                          "topk_successes": success, "raw_candidate_attempts": 5,
                          "raw_candidate_valid_connected_attempts": rawvalid,
                          "raw_candidate_protected_attempts": 2,
                          "selected_root_validity_fraction": .75,
                          "total_charged_seconds": 10, "post_update_charged_seconds": 4})
            for basis, budget in (("total", 10), ("post_update", 4)):
                frontier.append({"campaign_id": campaign, "method": method, "budget_basis": basis,
                                 "wall_budget_seconds": budget, "topk": 16,
                                 "topk_successes": success})
    audit = [{"campaign_id": campaign, "feasible_stratum": "true",
              "negative_goal_feasible_distinct": 1, "positive_goal_feasible_distinct": 1}
             for campaign in ("c0", "c1")]
    _csv(output / "cases.csv", cases)
    _csv(output / "budget_frontier.csv", frontier)
    _csv(output / "feasibility.csv", audit)
    _csv(output / "roots.csv", []) if False else None
    return task


def test_report_keeps_attempt_denominators_retries_and_paired_seed_clusters(tmp_path):
    study = tmp_path / "revision_recovery_001"
    tasks = []
    for seed in (142, 143, 144):
        task = _run(study, "test", seed, f"seed{seed}", delta=0)
        tasks.append(task)
    # A later retry for seed 143 is the resolved task; earlier data stay ignored.
    retry = _run(study, "test", 143, "seed143_retry1", delta=4)
    tasks[1] = retry
    (study / "tasks").mkdir(parents=True)
    (study / "tasks" / "recovery_test.resolved.json").write_text(json.dumps(tasks), encoding="utf-8")
    # A malformed operation folder is outside the reporter's input scope.
    operation = study / "operation" / "trap"
    operation.mkdir(parents=True)
    (operation / "summary.json").write_text("not json", encoding="utf-8")

    result = report.build_report(study)
    assert len(result["completed_runs"]) == 3
    assert any(row["path"].endswith("seed143_retry1") for row in result["completed_runs"])
    cases = result["case_summaries"]
    assert len([row for row in cases if row["method"] == "rollback_25" and row["stratum"] == "all"]) == 6
    row = next(row for row in cases if row["method"] == "rollback_25" and row["stratum"] == "all")
    assert row["raw_valid"] == 2
    assert row["raw_attempts"] == 5
    assert row["selected_root_validity"] == .75
    paired = [row for row in result["paired_campaign_bootstrap"]
              if row["method"] == "rollback_25" and row["baseline"] == "conditional_continue"
              and row["budget_basis"] == "total" and row["stratum"] == "all"]
    assert len(paired) == 1
    assert paired[0]["seeds"] == [142, 143, 144]
    assert paired[0]["paired_campaigns"] == 2
    assert paired[0]["bootstrap_samples"] == 2000
    assert paired[0]["mean_method_minus_baseline"] == (4 / 16 + 8 / 16 + 4 / 16) / 3
    assert paired[0]["status"] == "complete"
    assert (study / "live_report.md").is_file()
    assert (study / "analysis" / "report.json").is_file()
    assert {Path(name).suffix for name in result["figures"]} == {".png", ".pdf"}
    assert len(result["figures"]) == 4
