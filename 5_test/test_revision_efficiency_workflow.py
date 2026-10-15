"""Coordinator regressions isolated from live state and the GPU scheduler."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def workflow(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "3_model_training/scripts/revision_efficiency_workflow.py"
    spec = importlib.util.spec_from_file_location("revision_workflow_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STUDY", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "update", lambda state: None)
    monkeypatch.setattr(module, "report", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_conditional_completion_requires_checkpoint_and_full_training_budget(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    task = dict(stage="conditional", output=str(tmp_path / "model"), steps=10000)
    output = Path(task["output"])
    write(output / "training_summary.json", {"step": 10000})
    assert not module.complete(task)
    (output / "checkpoint.pt").write_bytes(b"checkpoint")
    assert module.complete(task)
    write(output / "training_summary.json", {"step": 9999})
    assert not module.complete(task)


def test_revision_completion_requires_budget_and_calibration_artifacts(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    task = dict(stage="revision_efficiency", name="test", seed=142, split="test",
                output=str(tmp_path / "evaluation"), wall_budgets_seconds=[1., 2.])
    output = Path(task["output"])
    write(output / "summary.json", dict(phase="complete", task=task))
    for name in ("attempts.csv", "cases.csv", "cost_batches.csv"):
        (output / name).write_text("header\n")
    assert not module.complete(task), "test stage must retain calibration and budget-frontier artifacts"
    (output / "calibration.csv").write_text("header\n")
    assert not module.complete(task)
    (output / "budget_frontier.csv").write_text("header\n")
    assert module.complete(task)
    assert not module.complete({**task, "wall_budgets_seconds": [1., 2., 4.]}), (
        "completed outputs belong to their recorded task configuration")


def test_completed_artifacts_skip_expired_scheduler_lookup(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    task = {"name": "finished"}
    monkeypatch.setattr(module, "complete", lambda task: True)
    monkeypatch.setattr(module, "gpu_hours", lambda jobs: 2.)
    monkeypatch.setattr(module, "command", lambda args: pytest.fail("completed task queried scheduler"))
    assert module.wait_job("expired", [task], {"jobs": ["expired"]}, "stage") == [task]


def test_scheduler_completion_waits_for_shared_filesystem_artifacts(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    visible = {"ready": False}
    task = {"name": "late_artifact"}
    monkeypatch.setattr(module, "complete", lambda item: visible["ready"])
    monkeypatch.setattr(module, "gpu_hours", lambda jobs: 0.)
    monkeypatch.setattr(module, "command", lambda args: "")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: visible.update(ready=True))
    assert module.wait_job("completed", [task], {"jobs": []}, "stage") == [task]


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "CANCELLED+"])
def test_expired_queue_job_uses_terminal_accounting_without_success_guess(
        tmp_path, monkeypatch, terminal):
    module = workflow(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "complete", lambda task: False)
    monkeypatch.setattr(module, "gpu_hours", lambda jobs: 0.)
    calls = []

    def command(args):
        calls.append(args[0])
        if args[0] == "squeue":
            raise subprocess.CalledProcessError(1, args)
        assert args[0] == "sacct"
        return terminal + "|\n"

    monkeypatch.setattr(module, "command", command)
    assert module.wait_job("expired", [{"name": "missing"}], {"jobs": []}, "stage") == []
    assert calls == ["squeue", "sacct"]


def test_reservation_ceiling_blocks_scheduler_submission(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "complete", lambda task: False)
    monkeypatch.setattr(module, "gpu_hours", lambda jobs: 47.)
    monkeypatch.setattr(module, "command", lambda args: pytest.fail("submitted over budget"))
    task = dict(name="new", stage="revision_efficiency", output=str(tmp_path / "new"))
    with pytest.raises(RuntimeError, match="(?i)reservation|48 GPU"):
        module.submit([task], "new", tmp_path / "snapshot", {"jobs": []}, minutes=120)


def test_partial_array_resume_keeps_original_mapping_and_does_not_submit(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    snapshot = tmp_path / "snapshot"
    tasks = [dict(name=f"task{i}", stage="revision_efficiency",
                  output=str(tmp_path / f"task{i}"), snapshot=str(snapshot)) for i in range(2)]
    array = tmp_path / "tasks" / "stage.json"
    module.atomic(array, tasks)
    original = array.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    state = dict(jobs=["original"], submissions=[dict(job="original", stage="stage",
                 tasks=[task["name"] for task in tasks], wall_minutes=90,
                 task_file=str(array), task_sha256=digest)])
    finished = {"task0"}
    monkeypatch.setattr(module, "complete", lambda task: task["name"] in finished)
    monkeypatch.setattr(module, "gpu_hours", lambda jobs: 0.)
    monkeypatch.setattr(module, "command", lambda args: pytest.fail("resume submitted duplicate GPU array"))

    def wait(job, pending, state, phase):
        assert job == "original"
        assert array.read_bytes() == original
        finished.update(task["name"] for task in pending)
        return list(pending)

    monkeypatch.setattr(module, "wait_job", wait)
    done = module.submit(tasks, "stage", snapshot, state, minutes=90)
    assert {task["name"] for task in done} == {"task0", "task1"}
    assert array.read_bytes() == original
    assert len(state["submissions"]) == 1


def test_budget_selection_uses_validation_timing_only(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    task = dict(split="validation", output=str(tmp_path / "validation"))
    path = Path(task["output"]) / "cost_batches.csv"
    path.parent.mkdir(parents=True)
    fields = ["campaign_id", "scenario", "method", "revision_target", "batch_first",
              "cumulative_charged_seconds", "topk_successes"]
    rows = [dict(zip(fields, ["case", "goal_reversal", "unguided_rerank", "", 0, .12, 0])),
            dict(zip(fields, ["case", "goal_reversal", "restart_control", .5, 0, 1., 0])),
            dict(zip(fields, ["case", "goal_reversal", "restart_control", .5, 8, 3.2, 0]))]

    def save():
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    save()
    first = module.select_budgets([task])
    for row in rows:
        row["topk_successes"] = 16
    save()
    assert module.select_budgets([task]) == first
    assert first == [.0625, .125, .25, .5, 1., 2., 4.]
    with pytest.raises(ValueError, match="validation"):
        module.select_budgets([{**task, "split": "test"}])


def test_unresolved_submission_intent_stops_without_duplicate_job(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    task = dict(name="task", stage="revision_efficiency", output=str(tmp_path / "output"))
    write(tmp_path / "tasks/stage.submission.json", dict(stage="stage", job=None))
    monkeypatch.setattr(module, "complete", lambda task: False)
    monkeypatch.setattr(module, "command", lambda args: pytest.fail("blindly retried submission"))
    with pytest.raises(RuntimeError, match="Unresolved submission intent"):
        module.submit([task], "stage", tmp_path / "snapshot", {"jobs": []})


def test_job_recorded_in_intent_recovers_without_new_submission(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    snapshot = tmp_path / "snapshot"
    task = dict(name="task", stage="revision_efficiency", output=str(tmp_path / "output"),
                snapshot=str(snapshot))
    array = tmp_path / "tasks/stage.json"
    write(array, [task])
    intent = dict(stage="stage", job="recovered", tasks=["task"], wall_minutes=90,
                  task_file=str(array), task_sha256=hashlib.sha256(array.read_bytes()).hexdigest())
    write(tmp_path / "tasks/stage.submission.json", intent)
    completed = set()
    monkeypatch.setattr(module, "complete", lambda task: task["name"] in completed)
    monkeypatch.setattr(module, "command", lambda args: pytest.fail("recovery submitted duplicate"))

    def wait(job, tasks, state, phase):
        assert job == "recovered"
        completed.update(task["name"] for task in tasks)
        return tasks

    monkeypatch.setattr(module, "wait_job", wait)
    state = {"jobs": [], "submissions": []}
    assert module.submit([task], "stage", snapshot, state) == [task]
    assert state["jobs"] == ["recovered"]
    assert state["submissions"] == [intent]


def ready_record(tmp_path):
    snapshot = tmp_path / "snapshot"
    source = snapshot / "emf/benchmark.py"
    source.parent.mkdir(parents=True)
    source.write_text("source = 'reviewed'\n")
    smoke = tmp_path / "smoke.json"
    write(smoke, {"phase": "complete"})
    report = tmp_path / "review.xml"
    report.write_text('<testsuites><testsuite tests="1" failures="0" errors="0"/></testsuites>')
    reference = lambda path: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(snapshot=str(snapshot), gpu_smoke_passed=True, review_tests_passed=True,
                source_sha256={"emf/benchmark.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                gpu_smoke_summary=reference(smoke), review_test_report=reference(report))


def test_readiness_binds_snapshot_and_review_evidence(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    ready = ready_record(tmp_path)
    assert module.validate_ready(ready) == Path(ready["snapshot"])
    Path(ready["snapshot"], "emf/benchmark.py").write_text("unreviewed change\n")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        module.validate_ready(ready)


def test_readiness_rejects_changed_evidence_and_incomplete_smoke(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    ready = ready_record(tmp_path)
    smoke = Path(ready["gpu_smoke_summary"]["path"])
    write(smoke, {"phase": "failed"})
    with pytest.raises(RuntimeError, match="evidence changed"):
        module.validate_ready(ready)
    ready["gpu_smoke_summary"]["sha256"] = hashlib.sha256(smoke.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="smoke did not complete"):
        module.validate_ready(ready)


def test_readiness_requires_both_cpu_and_gpu_checks(tmp_path, monkeypatch):
    module = workflow(tmp_path, monkeypatch)
    ready = ready_record(tmp_path)
    for field in ("gpu_smoke_passed", "review_tests_passed"):
        with pytest.raises(RuntimeError, match="CPU checks and allocated-GPU smoke"):
            module.validate_ready({**ready, field: False})
