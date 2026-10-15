"""Synthetic artifact checks for the bounded evidence-steering reporter."""

import csv
import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "emf_steering_report",
    Path(__file__).resolve().parents[1] / "3_model_training/scripts/steering_report.py",
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


FIELDS = (
    "source_seed", "source_variant", "condition", "support_size", "campaign", "family",
    "attempts", "local_profile_success_attempts", "raw_slot_local_profile_success_attempts",
    "profile_success_attempts", "raw_slot_profile_success_attempts",
    "local_profile_yield_per_attempt", "raw_slot_local_profile_yield_per_attempt",
    "profile_yield_per_attempt", "raw_slot_profile_yield_per_attempt", "steering_benefit",
)


def _write_case(root, rank, seed, support, *, pending=False):
    directory = root / f"factorized_rank{rank}" / f"seed_{seed}" / "evaluation_validation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps({
        "phase": "evaluation_complete" if not pending else "evaluation_pending",
        "support_sizes": [support], "aggregate": [],
    }))
    if pending:
        return
    rows = []
    for condition, benefit in (("correct", .1), ("opposite", .1)):
        rows.append({
            "source_seed": seed, "source_variant": "factorized", "condition": condition,
            "support_size": support, "campaign": "campaign_a", "family": "family_a",
            "attempts": 10, "local_profile_success_attempts": 2,
            "raw_slot_local_profile_success_attempts": 3,
            "profile_success_attempts": 3, "raw_slot_profile_success_attempts": 4,
            "local_profile_yield_per_attempt": .2,
            "raw_slot_local_profile_yield_per_attempt": .3,
            "profile_yield_per_attempt": .3, "raw_slot_profile_yield_per_attempt": .4,
            "steering_benefit": benefit,
        })
    metrics_path = directory / "metrics.csv"
    previous = []
    if metrics_path.exists():
        with metrics_path.open(newline="") as stream:
            previous = list(csv.DictReader(stream))
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([*previous, *rows])


def test_refresh_combines_two_seeds_and_supports_without_losing_rank_identity(tmp_path):
    root = tmp_path / "evidence_steering_001"
    root.mkdir()
    (root / "workflow_state.json").write_text(json.dumps({
        "phase": "evaluating", "launched_jobids": [7123, 7124],
    }))
    for rank in (8, 32):
        for seed in (42, 43):
            for support in (4, 16):
                _write_case(root, rank, seed, support)

    status = reporter.refresh(root)
    assert status == {"phase": "evaluating", "summaries": 4,
                      "metric_rows": 16, "missing_metrics": 0,
                      "preference_reversal_points": 0, "trace_stage_rows": 0}
    payload = json.loads((root / "analysis/summary.json").read_text())
    assert payload["metric_rows"] == 16
    assert payload["launched_jobids"] == [7123, 7124]
    assert len({row["directory_key"] for row in payload["evaluation_summaries"]}) == 4
    assert len({row["directory_key"].split("/")[0]
                for row in payload["evaluation_summaries"]}) == 2

    with (root / "analysis/combined_metrics.csv").open(newline="") as stream:
        combined = list(csv.DictReader(stream))
    assert len(combined) == 16
    assert {row["source_seed"] for row in combined} == {"42", "43"}
    assert {row["support_size"] for row in combined} == {"4", "16"}
    assert {row["directory_key"].split("/")[0] for row in combined} == {
        "factorized_rank8", "factorized_rank32"}

    report = (root / "analysis/report.md").read_text()
    assert "simulated measurements" in report
    assert "every attempted slot as the denominator" in report
    assert "Distinct local successes / yield" in report
    assert (root / "analysis/support_curves.png").stat().st_size > 100
    assert (root / "analysis/preference_reversal.png").stat().st_size > 100
    assert (root / "analysis/evidence_trace.png").stat().st_size > 100


def test_pending_evaluations_remain_pending_in_report_and_summary(tmp_path):
    root = tmp_path / "evidence_steering_pending"
    root.mkdir()
    (root / "workflow_state.json").write_text(json.dumps({
        "phase": "training", "launched": {
            "source": {"job_id": 9981, "state": "running"},
            "validation": {"job_id": 9982},
        },
    }))
    _write_case(root, 8, 42, 4, pending=True)

    status = reporter.refresh(root)
    assert status["summaries"] == 1 and status["metric_rows"] == 0
    assert status["preference_reversal_points"] == status["trace_stage_rows"] == 0
    payload = json.loads((root / "analysis/summary.json").read_text())
    assert payload["launched_jobids"] == {"source": 9981, "validation": 9982}
    assert payload["missing_metrics_paths"] == [
        "factorized_rank8/seed_42/evaluation_validation/metrics.csv"]
    assert payload["aggregates"] == []
    report = (root / "analysis/report.md").read_text()
    assert "no evaluation metric rows are available yet" in report
    assert "pending" in report
    assert "0.0000" not in report
    assert (root / "analysis/combined_metrics.csv").read_text() == ""
    assert (root / "analysis/support_curves.png").stat().st_size > 100
    assert (root / "analysis/preference_reversal.png").stat().st_size > 100
    assert (root / "analysis/evidence_trace.png").stat().st_size > 100


def test_support_curves_use_test_models_and_average_campaigns_then_seeds():
    rows = [
        {"split": "test", "directory_key": "source/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "a", "family": "f1",
         "local_profile_yield_per_attempt": .1},
        {"split": "test", "directory_key": "source/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "b", "family": "f2",
         "local_profile_yield_per_attempt": .3},
        {"split": "test", "directory_key": "source/r8_seed143", "source_seed": 143,
         "support_size": 16, "condition": "correct", "campaign": "a", "family": "f1",
         "local_profile_yield_per_attempt": .4},
        {"split": "test", "directory_key": "source/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "opposite", "campaign": "a", "family": "f1",
         "local_profile_yield_per_attempt": .2},
        {"split": "validation", "directory_key": "source/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "a", "family": "f1",
         "local_profile_yield_per_attempt": .9},
        {"split": "test", "directory_key": "operation/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "a", "family": "f1",
         "local_profile_yield_per_attempt": .8},
        {"split": "test", "directory_key": "privileged/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "a", "family": "f1",
         "local_profile_yield_per_attempt": .7},
    ]
    assert reporter._curve_model_key("source/r8_seed142") == "source/r8"
    means = reporter._support_curve_means(rows, "local_profile_yield_per_attempt")
    assert abs(means[("source/r8", 16.0, "correct")] - .3) < 1e-12
    assert abs(means[("source/r8", 16.0, "opposite")] - .2) < 1e-12
    assert len(means) == 2


def test_preference_reversal_uses_declared_pair_signs_and_correct_evidence_only():
    rows = [
        {"split": "test", "directory_key": "source/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "test_f0_p00_a",
         "family": 0, "mean_generated_u_change": .2},
        {"split": "test", "directory_key": "source/r8_seed142", "source_seed": 142,
         "support_size": 16, "condition": "correct", "campaign": "test_f0_p00_b",
         "family": 0, "mean_generated_u_change": -.1},
        {"split": "test", "directory_key": "source/r8_seed143", "source_seed": 143,
         "support_size": 16, "condition": "correct", "campaign": "test_f0_p00_a",
         "family": 0, "mean_generated_u_change": .4},
        {"split": "test", "directory_key": "source/r8_seed143", "source_seed": 143,
         "support_size": 16, "condition": "correct", "campaign": "test_f0_p00_b",
         "family": 0, "mean_generated_u_change": -.2},
        {"split": "test", "directory_key": "source/r8_seed143", "source_seed": 143,
         "support_size": 16, "condition": "opposite", "campaign": "test_f0_p00_b",
         "family": 0, "mean_generated_u_change": -.8},
    ]
    means = reporter._preference_reversal_means(rows)
    assert abs(means[("source/r8", 16.0, 1)] - .3) < 1e-12
    assert abs(means[("source/r8", 16.0, -1)] - -.15) < 1e-12


def test_trace_rows_use_emitted_stages_schema_and_plot_nonzero_values(tmp_path):
    trace = tmp_path / "trace/case_000"
    trace.mkdir(parents=True)
    (trace / "stages.json").write_text(json.dumps([
        {"time": .1, "comparison": "n128-n16", "correction_l2": 1e-8,
         "logit_delta_l2": .002},
        {"time": .5, "comparison": "n128-n16", "correction_l2": 2e-8,
         "logit_delta_l2": .003},
    ]))
    rows = reporter._trace_rows(tmp_path)
    assert len(rows) == 2
    reporter._plot_trace(tmp_path / "trace.png", rows)
    assert (tmp_path / "trace.png").stat().st_size > 100
