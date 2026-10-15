"""Bounded, restartable study coordinator; run this process in tmux."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(os.environ.get("EMF_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
STUDY = ROOT / "4_experimental_result/result/revision_efficiency_001"
PYTHON = "/hopper/home/miyao/miniconda3/envs/emf/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()


def complete(task):
    output = Path(task["output"])
    if task["stage"] == "conditional":
        path = output / "training_summary.json"
        return (output / "checkpoint.pt").exists() and path.exists() and read(path).get("step") == task["steps"]
    if task["stage"] == "critic":
        return (output / "critic.pt").exists() and (output / "summary.json").exists()
    path = output / "summary.json"
    if not path.exists() or read(path).get("phase") != "complete":
        return False
    recorded = read(path).get("task", {})
    if any(recorded.get(key) != value for key, value in task.items()):
        return False
    if task["stage"] == "revision_efficiency":
        expected = ["attempts.csv", "cases.csv", "cost_batches.csv", "calibration.csv"]
        if task.get("wall_budgets_seconds"):
            expected.append("budget_frontier.csv")
        return all((output / name).exists() for name in expected)
    return (output / "metrics.csv").exists()


def gpu_hours(jobs):
    operation = STUDY / "operation_manifest.json"
    if operation.exists():
        jobs = list(dict.fromkeys([*jobs, read(operation)["job"]]))
    # A parallel stage can finish before the coordinator reaches its wait.
    # Submission intents remain the authoritative record of allocated jobs.
    for intent_path in (STUDY / "tasks").glob("*.submission.json"):
        job = read(intent_path).get("job")
        if job and job not in jobs:
            jobs = [*jobs, job]
    if not jobs:
        return 0.
    lines = command(["sacct", "-n", "-X", "-P", "-j", ",".join(map(str, jobs)),
                     "--format=JobIDRaw,ElapsedRaw,AllocTRES"])
    seconds = 0
    for line in lines.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or "." in parts[0]:
            continue
        gpus = next((int(item.split("=")[1]) for item in parts[2].split(",")
                     if item.startswith("gres/gpu=")), 0)
        seconds += int(parts[1] or 0) * gpus
    return seconds / 3600


def update(state):
    state["updated_utc"] = now()
    atomic(STUDY / "status.json", state)
    begin, end = "<!-- REVISION_EFFICIENCY_001_BEGIN -->", "<!-- REVISION_EFFICIENCY_001_END -->"
    block = "\n".join([
        begin, "## Revision efficiency — next-stage study", "",
        f"Updated {state['updated_utc']}. Stage: **{state['phase']}**.", "",
        "This study uses the saved hard-masked B generator. It compares terminal control, ordinary conditional generation, expanded sampling and masked mutation. Revision tests distinguish more measurements from a reversed improvement request, with the second endpoint retained.", "",
        "[Plan](../detail_operation/06_revision_efficiency_experiment_plan.md) · "
        "[Live measurements](../../4_experimental_result/result/revision_efficiency_001/live_report.md) · "
        "[Task and scheduler records](../../4_experimental_result/result/revision_efficiency_001/)", "",
        f"Allocated GPU hours recorded at the last stage boundary: {state.get('allocated_gpu_hours', 0):.2f}. "
        "A running stage is not a completed result.", "",
        ("Recorded operational failures: " + "; ".join(state["errors"])) if state.get("errors") else "",
        "", end, ""])
    path = ROOT / "1_research_file/result_log/experiment_results_1014.md"
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = path.read_text() if path.exists() else "# Experiment results — 1014\n"
        if begin in old:
            before, rest = old.split(begin, 1)
            _, after = rest.split(end, 1)
            new = before + block + after.lstrip("\n")
        else:
            new = old.rstrip() + "\n\n" + block
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(new)
        temporary.replace(path)


def report():
    path = ROOT / "3_model_training/scripts/revision_efficiency_report.py"
    if path.exists():
        result = subprocess.run([PYTHON, str(path), "--study", str(STUDY)], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            print("Report refresh failed:", result.stderr[-1500:], flush=True)


def wait_job(job, tasks, state, phase):
    state["phase"] = phase
    update(state)
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    while not all(complete(task) for task in tasks):
        try:
            active = command(["squeue", "-h", "-j", str(job), "-o", "%i|%T"])
        except subprocess.CalledProcessError:
            records = command(["sacct", "-n", "-X", "-P", "-j", str(job), "--format=State"])
            states = [line.split("|")[0].split()[0].rstrip("+") for line in records.splitlines() if line.strip()]
            if states and all(value in terminal for value in states):
                active = ""
            else:
                raise
        if not active:
            # Shared-filesystem output visibility can lag Slurm completion.
            # Give completed writes time to appear before scheduling a retry.
            for _ in range(9):
                if all(complete(task) for task in tasks):
                    break
                time.sleep(10)
            break
        state["completed_tasks"] = [task["name"] for task in tasks if complete(task)]
        update(state)
        time.sleep(30)
    state["allocated_gpu_hours"] = gpu_hours(state.get("jobs", []))
    update(state)
    report()
    return [task for task in tasks if complete(task)]


def submit(tasks, label, snapshot, state, minutes=90):
    pending = [task for task in tasks if not complete(task)]
    array = STUDY / "tasks" / f"{label}.json"
    intent_path = STUDY / "tasks" / f"{label}.submission.json"
    previous = next((entry for entry in state.get("submissions", [])
                     if entry["stage"] == label), None)
    if previous is None and intent_path.exists():
        previous = read(intent_path)
        if not previous.get("job"):
            raise RuntimeError(f"Unresolved submission intent: {intent_path}; inspect scheduler before resubmitting")
        state.setdefault("submissions", []).append(previous)
        if previous["job"] not in state.setdefault("jobs", []):
            state["jobs"].append(previous["job"])
        update(state)
    if not pending:
        return tasks
    if previous:
        # Workers use their array index to read this file. Never rewrite an
        # existing submission when a completed subset disappears from pending.
        array = Path(previous.get("task_file", array))
        if hashlib.sha256(array.read_bytes()).hexdigest() != previous["task_sha256"]:
            raise RuntimeError(f"Submitted task array changed: {array}")
        submitted = read(array)
        by_name = {task["name"]: task for task in submitted}
        for task in pending:
            if task["name"] not in by_name or {**task, "snapshot": str(snapshot)} != by_name[task["name"]]:
                raise RuntimeError(f"Task changed on resume: {task['name']}")
        job = previous["job"]
    else:
        used = gpu_hours(state.get("jobs", []))
        if used + len(pending) * minutes / 60 > 48:
            raise RuntimeError(f"Study reservation exceeds 48 GPU hours: used={used:.2f}, new ceiling={len(pending)*minutes/60:.2f}")
        for task in pending:
            task["snapshot"] = str(snapshot)
        if array.exists() and read(array) != pending:
            raise RuntimeError(f"Refusing to replace an existing task array: {array}")
        atomic(array, pending)
        intent = dict(job=None, stage=label, tasks=[task["name"] for task in pending],
                      wall_minutes=minutes, task_file=str(array),
                      task_sha256=hashlib.sha256(array.read_bytes()).hexdigest(), submitted_utc=now())
        atomic(intent_path, intent)
        job = command(["sbatch", "--parsable", f"--time={minutes//60:02d}:{minutes%60:02d}:00",
                       f"--array=0-{len(pending)-1}%6", f"--output={STUDY}/logs/{label}_%A_%a.log",
                       str(snapshot / "scripts/revision_efficiency_gpu.slurm"), str(array), str(snapshot)]).split(";")[0]
        state.setdefault("jobs", []).append(job)
        intent["job"] = job
        atomic(intent_path, intent)
        state.setdefault("submissions", []).append(intent)
        update(state)
    done = [task for task in tasks if complete(task)] + wait_job(job, pending, state, label)
    failed = [task for task in pending if not complete(task)]
    if failed:
        state.setdefault("errors", []).append(f"{label}: incomplete tasks {[task['name'] for task in failed]}")
        update(state)
        # A fresh path preserves partial attempts and makes the bounded retry visible.
        retries = [dict(task, name=task["name"] + "_retry1", output=task["output"] + "_retry1",
                        retry_attempt=1) for task in failed if not task.get("retry_attempt")]
        if retries:
            done.extend(submit(retries, label + "_retry1", snapshot, state, minutes))
    return done


def require(tasks, count, label):
    if len(tasks) != count:
        raise RuntimeError(f"{label}: {len(tasks)}/{count} tasks completed; inspect retained logs")
    return tasks


def validate_ready(ready):
    """Bind the exact code snapshot to completed CPU and allocated GPU checks."""
    snapshot = Path(ready["snapshot"])
    if not ready.get("gpu_smoke_passed") or not ready.get("review_tests_passed"):
        raise RuntimeError("revision evaluator requires reviewed CPU checks and allocated-GPU smoke")
    hashes = ready.get("source_sha256", {})
    if not hashes:
        raise RuntimeError("reviewed snapshot hashes are missing")
    for relative, expected in hashes.items():
        path = snapshot / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Reviewed snapshot changed: {relative}")
    for name in ("gpu_smoke_summary", "review_test_report"):
        record = ready[name]
        path = Path(record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Review evidence changed: {name}")
    if read(ready["gpu_smoke_summary"]["path"]).get("phase") != "complete":
        raise RuntimeError("allocated-GPU smoke did not complete")
    return snapshot


def critic_audit_score(task):
    values = read(Path(task["output"]) / "summary.json").get("diagnostics", {}).get("audit_generated", {}).get("rmse_scaled")
    if not values or not all(math.isfinite(value) for value in values):
        return float("inf"), True
    return sum(values), any(value > .2 for value in values)


def select_budgets(tasks):
    """Choose a logarithmic time grid from validation timing, never outcomes."""
    import numpy as np
    first, totals = [], []
    sources = []
    for task in tasks:
        if task["split"] != "validation":
            raise ValueError("budget selection requires validation timing")
        path = Path(task["output"]) / "cost_batches.csv"
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        cases = {}
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row["campaign_id"], row["scenario"], row["method"], row["revision_target"])
                value = float(row["cumulative_charged_seconds"])
                if row["method"] == "unguided_rerank" and int(row["batch_first"]) == 0:
                    first.append(value)
                if row["method"] == "restart_control":
                    cases[key] = max(value, cases.get(key, 0.))
        totals.extend(cases.values())
    if not first or not totals or not all(math.isfinite(x) and x > 0 for x in first + totals):
        raise ValueError("validation timing is missing or nonfinite")
    lower = max(0.01, float(np.median(first)))
    upper = max(lower * 8, float(np.median(totals)))
    lo, hi = math.floor(math.log2(lower)), math.ceil(math.log2(upper))
    budgets = [float(2. ** power) for power in range(lo, hi + 1)]
    decision = dict(selection_split="validation", selected_utc=now(), wall_budgets_seconds=budgets,
                    rule="Powers of two from median first unguided batch (rounded down) to median final controlled-restart cost (rounded up); timing only, no success inputs.",
                    validation_paths=sources, median_first_unguided=lower,
                    median_final_restart=float(np.median(totals)),
                    cap_policy="64 controlled and 1600 conditional/unguided proposals per case on test; explicitly mark exhausted caps.")
    atomic(STUDY / "budget_selection.json", decision)
    return budgets


def main():
    state = read(STUDY / "status.json")
    manifest = read(STUDY / "manifest.json")
    prep = manifest["preparation_tasks"]
    done = wait_job(state["preparation_job"], prep, state, "source_preparation")
    if len(done) != len(prep):
        missing = [task for task in prep if not complete(task)]
        done += submit([dict(task, name=task["name"] + "_retry1", output=task["output"] + "_retry1", retry_attempt=1)
                        for task in missing], "preparation_retry1", Path(manifest["source_snapshot"]), state, 60)
    require(done, 6, "source preparation")
    resolution_path = STUDY / "preparation_resolution.json"
    if resolution_path.exists():
        done = read(resolution_path)
        require([task for task in done if complete(task)], 6, "frozen preparation selection")
        resolved = {(task["seed"], task["stage"]): task for task in done}
    else:
        resolved = {(task["seed"], task["stage"]): task for task in done}
        # Same source-only critic repair rule used for the preceding A study.
        repairs = [dict(task, name=task["name"] + "_refined", output=task["output"] + "_refined",
                        steps=5000, source_attempts=1024)
                   for task in done if task["stage"] == "critic" and critic_audit_score(task)[1]]
        if repairs:
            refined = submit(repairs, "critic_refinement", Path(manifest["source_snapshot"]), state, 45)
            decisions = []
            for task in refined:
                key = (task["seed"], "critic")
                old = resolved[key]
                selected = task if critic_audit_score(task)[0] < critic_audit_score(old)[0] else old
                resolved[key] = selected
                decisions.append(dict(seed=task["seed"], old_audit_score=critic_audit_score(old)[0],
                                      refined_audit_score=critic_audit_score(task)[0], selected=selected))
            atomic(STUDY / "critic_selection.json", dict(rule="Lower sum of finite source-generated audit RMSE after one triggered refit; no test outcomes", decisions=decisions))
        done = list(resolved.values())
        atomic(resolution_path, done)
    static = []
    for seed in manifest["seeds"]:
        prior = ROOT / f"4_experimental_result/result/masked_control_001/source/prior_B_seed{seed}/checkpoint.pt"
        static.append(dict(stage="control_test", name=f"static_B_seed{seed}", seed=seed, variant="B",
            checkpoint=str(prior), critic=str(Path(resolved[(seed, "critic")]["output"]) / "critic.pt"),
            conditional_checkpoint=str(Path(resolved[(seed, "conditional")]["output"]) / "checkpoint.pt"),
            output=str(STUDY / "static_b" / f"seed{seed}"), split="test", support_sizes=[0, 4, 8, 16],
            attempts=64, chunk=8, nfe=16, iterations=4, radii=[.5, 2.], compute_pool_multiplier=25))
    require(submit(static, "static_b", Path(manifest["source_snapshot"]), state, 90), 3, "B comparison")
    state["phase"] = "awaiting_reviewed_revision_evaluator"
    update(state)
    while not (STUDY / "reviewed_ready.json").exists():
        time.sleep(30)
    ready = read(STUDY / "reviewed_ready.json")
    snapshot = validate_ready(ready)
    base = [dict(task, stage="revision_efficiency", support_sizes=[4, 16],
                 conditional_attempts=1600, baseline_attempts=1600,
                 revision_times=[.25, .5, .75], topk=16) for task in static]
    validation = [dict(task, name=f"revision_validation_seed{task['seed']}", split="validation", attempts=32,
                       output=str(STUDY / "validation" / f"seed{task['seed']}")) for task in base]
    validation = require(submit(validation, "revision_validation", snapshot, state, 120), 3, "revision validation")
    budgets = select_budgets(validation)
    tests = [dict(task, name=f"revision_test_seed{task['seed']}", split="test", attempts=64,
                  wall_budgets_seconds=budgets,
                  output=str(STUDY / "test" / f"seed{task['seed']}")) for task in base]
    require(submit(tests, "revision_test", snapshot, state, 180), 3, "revision test")
    state.update(phase="complete_with_recorded_failures" if state.get("errors") else "complete",
                 allocated_gpu_hours=gpu_hours(state.get("jobs", [])), completed_utc=now())
    update(state)
    report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    with (STUDY / "coordinator.lock").open("a") as coordinator_lock:
        fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as error:
            state = read(STUDY / "status.json") if (STUDY / "status.json").exists() else {}
            state.setdefault("errors", []).append(f"{type(error).__name__}: {error}")
            state["phase"] = "failed"
            update(state)
            raise
