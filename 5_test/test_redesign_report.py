"""Reporter checks for external truth, paired comparisons, locks and accounting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "emf_redesign_report",
    Path(__file__).resolve().parents[1] / "3_model_training/scripts/redesign_report.py",
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _case(root, directory="evaluation_factorized_42", *, success=0.25,
          outcome="exact_simulator", attempts=1000, architecture="factorized"):
    parent = root / directory / "evaluation_test"
    _json(parent / "protocol.json", {"eval_split": "test", "nfe": 32, "k": 16})
    path = parent / "case_000/support_016/native"
    metadata = {"source_variant": architecture, "source_seed": 42, "campaign": "test_000",
                "lead_id": "lead_000", "support_size": 16, "mode": "native",
                "common_scorer_checkpoint_sha256": "common-scorer",
                "generator_checkpoint_sha256": architecture + "-generator"}
    generation = {"attempts": attempts, "nfe": 32, "generation_seconds": 2.0,
                  "response_scoring_seconds": 0.2, "metadata": metadata,
                  "predicted_profile_score": 0.9999}
    _json(path / "generation_summary.json", generation)
    _json(path / "evaluation_summary.json", {
        "attempts": attempts, "support_size": 16, "allocated_selection_slots": 16,
        "success": success, "efficiency": success / 2,
        "pre_ranking_unmeasured_joint_profile_yield_per_attempt": 0.02,
        "pre_ranking_neighbourhood_joint_profile_yield_per_attempt": 0.01,
        "full_pool_outcome_reader_calls": 800, "evaluation_kind": outcome,
        "metadata": generation,
    })
    return path


def test_collector_uses_external_outcomes_and_separates_preliminary_runs(tmp_path):
    _case(tmp_path, success=0.0)
    _case(tmp_path, "pilot", success=0.5)
    _case(tmp_path, "operation", success=1.0, attempts=11)
    _case(tmp_path, "retrospective", outcome="recorded_library")
    rows, generations, _, _, ignored = reporter.collect_artifacts(tmp_path)
    assert len(rows) == 3 and len(generations) == 4
    assert {row["stage"] for row in rows} == {"controlled", "pilot", "operation"}
    controlled = next(row for row in rows if row["stage"] == "controlled")
    assert controlled["success"] == 0 and controlled["full_protocol"]
    assert len(ignored) == 1 and ignored[0]["outcome_source"] == "recorded_library"


def test_uncertainty_averages_independent_campaigns_before_lead_count():
    rows = [{"seed": 42, "campaign": "A", "success": 1.0} for _ in range(10)]
    rows += [{"seed": 42, "campaign": "B", "success": 0.0}]
    result = reporter.clustered_estimate(rows, "success")
    assert result["mean"] == 0.5
    assert result["campaigns"] == 2 and result["seeds"] == 1
    assert result["ci_low"] == 0 and result["ci_high"] == 1
    single = reporter.clustered_estimate(rows[:10], "success")
    assert single["ci_low"] is None and single["ci_high"] is None


def test_architecture_pairing_requires_the_same_native_scorer(tmp_path):
    _case(tmp_path, success=0.25)
    _case(tmp_path, "evaluation_conditional_42", success=0.125, architecture="conditional")
    rows, *_ = reporter.collect_artifacts(tmp_path)
    pairs, curves, mismatches = reporter.paired_differences(rows)
    assert len(pairs) == len(curves) == 1 and not mismatches
    assert pairs[0]["success"] == -0.125
    other = dict(next(row for row in rows if row["architecture"] == "conditional"))
    other["common_scorer_checkpoint_sha256"] = "other-scorer"
    _, _, mismatches = reporter.paired_differences([*rows, other])
    assert len(mismatches) == 1


def test_marked_journal_edits_preserve_both_manual_regions(tmp_path):
    path = tmp_path / "experiment_results_1011.md"
    path.write_text("User introduction.\n")
    reporter.update_marked_section(path, "Generated first pass.")
    with path.open("a") as stream:
        stream.write("\nCoordinator's later note.\n")
    reporter.update_marked_section(path, "Generated second pass.")
    value = path.read_text()
    assert value.startswith("User introduction.")
    assert value.endswith("Coordinator's later note.\n")
    assert "Generated first pass." not in value and "Generated second pass." in value
    assert Path(str(path) + ".lock").exists()


def test_slurm_queries_only_tracked_ids_and_are_throttled(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            "123|source|COMPLETED|0:0|1800|gres/gpu=1,gres/gpu:l40s=1|start|end|node|123\n"
            "123.batch|batch|COMPLETED|0:0|1800|gres/gpu=1|start|end|node|123.batch\n"
            "999|other|RUNNING|0:0|3600|gres/gpu=1|start|end|node|999\n"
        ))

    monkeypatch.setattr(reporter.subprocess, "run", run)
    state = {"jobs": [123]}
    first = reporter.refresh_accounting(tmp_path, state, now=1000)
    assert len(first["jobs"]) == 1 and first["jobs"][0]["allocated_gpu_hours"] == 0.5
    reporter.refresh_accounting(tmp_path, state, now=1059)
    assert len(calls) == 1
    reporter.refresh_accounting(tmp_path, state, now=1060)
    assert len(calls) == 2
    assert calls[0][calls[0].index("--jobs") + 1] == "123"


def test_once_creates_analysis_and_only_marked_daily_sections(tmp_path):
    root = tmp_path / "4_experimental_result/result/calibrated_redesign_001"
    journal = tmp_path / "1_research_file/result_log"
    journal.mkdir(parents=True)
    for name in ("experiment_results_1011.md", "implementation_log_1011.md"):
        (journal / name).write_text("Manual record.\n")
    northstar = tmp_path / "1_research_file/research_plan.md"
    northstar.write_text("Preserved research direction.\n")
    _json(root / "workflow_state.json", {"phase": "tests complete", "jobs": []})
    _case(root)
    source = root / "source_factorized_42"
    _json(source / "manifest.json", {
        "config": {"redesign": {"variant": "factorized"}, "seed": 42},
        "parameter_count": 5355397, "trainable_parameters": 4000000,
    })
    _json(source / "source_summary.json", {
        "phase": "source_complete", "step": 10, "training_seconds": 20,
        "frozen_representation_unchanged": True,
    })
    snapshot = reporter.run_once(root, tmp_path, journal, slurm=False, plots=False)
    assert snapshot["controlled_batches"] == snapshot["controlled_test_batches"] == 1
    assert snapshot["sources"][0]["parameter_count"] == 5355397
    assert northstar.read_text() == "Preserved research direction.\n"
    for name in ("experiment_results_1011.md", "implementation_log_1011.md"):
        content = (journal / name).read_text()
        assert content.startswith("Manual record.\n") and reporter.BEGIN in content
    assert (root / "analysis/case_metrics.csv").exists()
    assert (root / "analysis/report.json").exists()


def test_plotting_uses_observed_curves_and_skips_empty_data(tmp_path):
    pytest.importorskip("matplotlib")
    assert reporter.make_plots(tmp_path, [], [], []) == []
    rows = [{"stage": "controlled", "eval_split": "test", "architecture": "factorized",
             "mode": "native", "support_size": 16, "success": 0.25,
             "efficiency": 0.125, "pre_ranking_neighbourhood_joint_profile_yield_per_attempt": 0.01}]
    names = reporter.make_plots(tmp_path, rows, [], [])
    assert names == ["support_curves_test.png"] and (tmp_path / names[0]).is_file()
