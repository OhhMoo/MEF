"""Study-level gates must use the declared split and paired candidate budgets."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "3_model_training/scripts/revision_recovery_workflow.py"
SPEC = importlib.util.spec_from_file_location("recovery_workflow_checks", PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def _validation(tmp_path, *, direct=2, adjusted=3, adjusted_valid=8):
    output = tmp_path / "validation"
    output.mkdir()
    records = []
    for cid in ("a", "b"):
        for method, successes, valid in (("conditional_continue", direct, 10),
                                         ("adapter_continue", adjusted, adjusted_valid)):
            records.append(dict(campaign_id=cid, method=method, topk=16,
                                topk_successes=successes,
                                valid_connected_attempts=valid, attempts=40))
    with (output / "cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return [dict(output=str(output), seed=142, split="validation")]


def test_adapter_gate_requires_success_gain_and_declared_validity_tolerance(tmp_path):
    gate = dict(min_success_gain=0., max_validity_drop=.05)
    tasks = _validation(tmp_path, adjusted_valid=9)
    result = workflow.adapter_gate(tasks, gate)
    assert result["passed"] and result["paired_cases"] == 2
    assert result["success_gain"] == 1/16
    assert result["validity_change"] == pytest.approx(-.025)
    tasks[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        workflow.adapter_gate(tasks, gate)


def test_adapter_gate_rejects_validity_damage_and_no_success_gain(tmp_path):
    first = tmp_path / "damage"
    first.mkdir()
    tasks = _validation(first, adjusted_valid=7)
    assert not workflow.adapter_gate(tasks, dict(min_success_gain=0., max_validity_drop=.05))["passed"]
    second = tmp_path / "identity"
    second.mkdir()
    tasks = _validation(second, adjusted=2, adjusted_valid=10)
    assert not workflow.adapter_gate(tasks, dict(min_success_gain=0., max_validity_drop=.05))["passed"]


def test_completion_marker_binds_exact_task_and_required_artifacts(tmp_path):
    task = dict(stage="adapter_train", output=str(tmp_path), seed=142, steps=500)
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "training_summary.json").write_text('{}')
    marker = tmp_path / "task_complete.json"
    marker.write_text(json.dumps({"task_sha256": workflow.task_digest(task)}))
    assert workflow.complete(task)
    assert not workflow.complete(dict(task, steps=501))
    (tmp_path / "checkpoint.pt").unlink()
    assert not workflow.complete(task)
