"""Bounded overnight coordinator for rollback, recovery, and one revision net.

Submit at most two three-seed stages concurrently. Source recovery decides
whether to train the adapter; exact validation decides whether to evaluate it
on the previously examined test campaigns. No test result changes either gate.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


ROOT = Path(os.environ.get("EMF_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PYTHON = "/hopper/home/miyao/miniconda3/envs/emf/bin/python"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_digest(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def complete(task):
    output = Path(task["output"])
    record = output / "task_complete.json"
    if not record.exists() or read(record).get("task_sha256") != task_digest(task):
        return False
    if task["stage"] == "adapter_train":
        return (output / "checkpoint.pt").exists() and (output / "training_summary.json").exists()
    required = ("summary.json", "cases.csv", "attempts.csv", "roots.csv",
                "budget_frontier.csv", "feasibility.csv", "reference_witnesses.csv")
    return all((output / name).exists() for name in required) and read(output / "summary.json").get("phase") == "complete"


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return str(value).lower() in {"true", "1"}


def source_gate(tasks, gate):
    """Positive evidence only, from source roots in the declared intervention ball."""
    recovered, campaigns = set(), set()
    sources = []
    for task in tasks:
        if task["split"] != "train":
            raise ValueError("Recovery training gate accepts source campaigns only")
        path = Path(task["output"]) / "roots.csv"
        sources.append({"path": str(path), "sha256": digest(path)})
        for row in rows(path):
            radius = float(row["bounded_radius_rms"])
            direct = truth(row["direct_success"])
            found = truth(row["randomized_suffix_any_success"]) or truth(row["oracle_adaptive_any_success"])
            if not direct and found and radius <= gate["max_delta_rms"] + 1e-8:
                key = (row["campaign_id"], int(row["seed"]), int(row["root_sample"]))
                recovered.add(key)
                campaigns.add(row["campaign_id"])
    return {"passed": len(recovered) >= gate["min_recovered_roots"] and len(campaigns) >= gate["min_campaigns"],
            "recovered_roots": len(recovered), "campaigns": sorted(campaigns),
            "criteria": gate, "sources": sources, "split": "train", "decided_utc": now()}


def adapter_gate(tasks, gate):
    """Evaluate the one fixed learned map once, using exact validation outcomes."""
    totals = {name: {"success": 0, "slots": 0, "valid": 0, "attempts": 0}
              for name in ("conditional_continue", "adapter_continue")}
    sources = []
    paired = {}
    for task in tasks:
        if task["split"] != "validation":
            raise ValueError("Adapter acceptance requires validation campaigns")
        path = Path(task["output"]) / "cases.csv"
        sources.append({"path": str(path), "sha256": digest(path)})
        for row in rows(path):
            method = row["method"]
            if method not in totals:
                continue
            values = totals[method]
            values["success"] += int(row["topk_successes"])
            values["slots"] += int(row["topk"])
            values["valid"] += int(row["valid_connected_attempts"])
            values["attempts"] += int(row["attempts"])
            paired.setdefault((row["campaign_id"], int(task["seed"])), set()).add(method)
    if not paired or any(len(names) != 2 for names in paired.values()):
        raise ValueError("Adapter gate lacks complete paired validation cases")
    direct, adapter = totals["conditional_continue"], totals["adapter_continue"]
    if direct["slots"] != adapter["slots"] or direct["attempts"] != adapter["attempts"]:
        raise ValueError("Adapter gate requires equal candidate and slot budgets")
    success_gain = (adapter["success"] - direct["success"]) / direct["slots"]
    validity_change = adapter["valid"] / adapter["attempts"] - direct["valid"] / direct["attempts"]
    return {"passed": success_gain > gate["min_success_gain"] and validity_change >= -gate["max_validity_drop"],
            "success_gain": success_gain, "validity_change": validity_change,
            "totals": totals, "paired_cases": len(paired), "criteria": gate,
            "sources": sources, "split": "validation", "decided_utc": now()}


class Coordinator:
    def __init__(self, study):
        self.study = Path(study).resolve()
        self.manifest = read(self.study / "manifest.json")
        self.snapshot = Path(self.manifest["snapshot"])
        self.state = read(self.study / "status.json") if (self.study / "status.json").exists() else {
            "phase": "created", "started_utc": now(), "jobs": [], "events": [], "errors": []}
        self.last_report_signature = None

    def update(self, phase=None, **values):
        if phase is not None:
            self.state["phase"] = phase
        self.state.update(values, updated_utc=now())
        atomic(self.study / "status.json", self.state)

    def validate_ready(self):
        ready = read(self.study / "reviewed_ready.json")
        if not ready.get("cpu_passed") or not ready.get("gpu_passed"):
            raise RuntimeError("Reviewed CPU and allocated-GPU operation must pass before full runs")
        for name, expected in self.manifest["snapshot_sha256"].items():
            if digest(self.snapshot / name) != expected:
                raise RuntimeError(f"Frozen source changed: {name}")
        for item in ready["evidence"]:
            if digest(item["path"]) != item["sha256"]:
                raise RuntimeError(f"Review artifact changed: {item['path']}")
        if ready.get("snapshot_sha256") != self.manifest["snapshot_sha256"]:
            raise RuntimeError("Review is for another source snapshot")
        for item in self.manifest["input_provenance"]:
            if digest(item["path"]) != item["sha256"]:
                raise RuntimeError(f"Pinned input changed: {item['path']}")

    def submit(self, label, tasks, minutes):
        resolution = self.study / "tasks" / f"{label}.resolved.json"
        if resolution.exists():
            done = read(resolution)
            if len(done) == len(tasks) and all(complete(task) for task in done):
                return {"label": label, "tasks": done, "minutes": minutes, "job": None}
        path = self.study / "tasks" / f"{label}.json"
        if path.exists() and read(path) != tasks:
            raise RuntimeError(f"Immutable tasks changed: {label}")
        if not path.exists():
            atomic(path, tasks)
        intent = self.study / "tasks" / f"{label}.submission.json"
        if intent.exists():
            old = read(intent)
            if old["task_sha256"] != digest(path):
                raise RuntimeError("Submitted task manifest changed")
            job = old["job"]
        else:
            (self.study / "logs").mkdir(exist_ok=True)
            args = ["sbatch", "--parsable", f"--array=0-{len(tasks)-1}%3",
                    f"--time={minutes}", f"--job-name=emf-{label[:18]}",
                    f"--output={self.study}/logs/{label}_%A_%a.log",
                    str(self.snapshot / "scripts/revision_recovery_gpu.slurm"), str(path), str(self.snapshot)]
            job = subprocess.check_output(args, cwd=ROOT, text=True).strip().split(";")[0]
            atomic(intent, {"job": job, "task_sha256": digest(path), "submitted_utc": now(), "args": args})
        if job not in self.state["jobs"]:
            self.state["jobs"].append(job)
        self.state["events"].append({"stage": label, "job": job, "utc": now()})
        self.update()
        return {"label": label, "tasks": tasks, "minutes": minutes, "job": job}

    def active(self, job):
        result = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%i|%T"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return True
        states = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(job), "--format=State"], capture_output=True, text=True)
        values = [line.split("|")[0].split()[0].rstrip("+") for line in states.stdout.splitlines() if line.strip()]
        return not (values and all(value in TERMINAL for value in values))

    def refresh_report(self):
        script = self.snapshot / "scripts/revision_recovery_report.py"
        if script.exists():
            artifacts = [p for group in self.manifest["task_groups"]
                         for p in (self.study / group).glob("*/task_complete.json")]
            artifacts += [p for p in (self.study / "source_gate.json", self.study / "adapter_gate.json") if p.exists()]
            signature = (self.state.get("phase"), tuple(sorted((str(p), p.stat().st_mtime_ns) for p in artifacts)))
            if signature == self.last_report_signature:
                return
            try:
                result = subprocess.run([PYTHON, str(script), "--study", str(self.study)], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                print("Report refresh exceeded 120 seconds; scientific jobs continue.", flush=True)
                return
            if result.returncode:
                print("Report:", result.stderr[-1200:], flush=True)
            else:
                self.last_report_signature = signature

    def wait(self, handle):
        tasks, label = handle["tasks"], handle["label"]
        self.update(label)
        if handle["job"] is not None:
            settled = None
            while not all(complete(task) for task in tasks):
                live = self.active(handle["job"])
                if not live:
                    settled = settled or time.monotonic()
                    if time.monotonic() - settled >= 90:
                        break
                self.update(completed_stage_tasks=[t["name"] for t in tasks if complete(t)])
                self.refresh_report()
                time.sleep(30)
        done = [task for task in tasks if complete(task)]
        missing = [task for task in tasks if not complete(task)]
        if missing:
            self.state["errors"].append({"stage": label, "missing": [t["name"] for t in missing], "utc": now()})
            if any(task.get("retry_attempt") for task in missing):
                raise RuntimeError(f"{label}: failed after bounded retry")
            retries = [dict(t, name=t["name"]+"_retry1", output=t["output"]+"_retry1", retry_attempt=1) for t in missing]
            done += self.wait(self.submit(label+"_retry1", retries, handle["minutes"]))
        done.sort(key=lambda task: int(task["seed"]))
        atomic(self.study / "tasks" / f"{label}.resolved.json", done)
        self.refresh_report()
        return done

    def tasks(self, group):
        return self.manifest["task_groups"][group]

    def run(self):
        self.validate_ready()
        source_job = self.submit("source_recovery", self.tasks("source"), 120)
        validation_job = self.submit("recovery_validation", self.tasks("validation"), 240)
        source = self.wait(source_job)
        gate = source_gate(source, self.manifest["source_gate"])
        atomic(self.study / "source_gate.json", gate)
        train_job = self.submit("adapter_train", self.tasks("adapter_train"), 120) if gate["passed"] else None
        if not gate["passed"]:
            self.update(adapter_decision="Skipped: source recoverability threshold not met")
        self.wait(validation_job)
        test_job = self.submit("recovery_test", self.tasks("test"), 360)
        adapter_test_job = None
        if train_job is not None:
            trained = self.wait(train_job)
            checkpoints = {task["seed"]: str(Path(task["output"]) / "checkpoint.pt") for task in trained}
            val_tasks = [dict(task, adapter_checkpoint=checkpoints[task["seed"]]) for task in self.tasks("adapter_validation")]
            val = self.wait(self.submit("adapter_validation", val_tasks, 180))
            decision = adapter_gate(val, self.manifest["adapter_gate"])
            atomic(self.study / "adapter_gate.json", decision)
            if decision["passed"]:
                test_tasks = [dict(task, adapter_checkpoint=checkpoints[task["seed"]]) for task in self.tasks("adapter_test")]
                adapter_test_job = self.submit("adapter_test", test_tasks, 240)
            else:
                self.update(adapter_decision="Fixed adapter failed exact validation gate; no further architecture tuning")
        self.wait(test_job)
        if adapter_test_job:
            self.wait(adapter_test_job)
        self.update("complete_with_recorded_retries" if self.state["errors"] else "complete", completed_utc=now())
        self.refresh_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=str(ROOT / "4_experimental_result/result/revision_recovery_001"))
    args = parser.parse_args()
    study = Path(args.study)
    with (study / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinator = Coordinator(study)
        try:
            coordinator.run()
        except Exception as error:
            coordinator.state["errors"].append({"error": f"{type(error).__name__}: {error}", "utc": now()})
            coordinator.update("failed")
            coordinator.refresh_report()
            raise


if __name__ == "__main__":
    main()
