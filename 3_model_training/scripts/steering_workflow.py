"""Prepare, launch and supervise the bounded overnight evidence-steering study."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "4_experimental_result/result/evidence_steering_001"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    sys.path.insert(0, str(ROOT / "3_model_training"))
    from emf.redesign_train import RedesignRunConfig
    from emf.redesign_contracts import RedesignConfig
    if (STUDY / "study_manifest.json").exists():
        raise FileExistsError("study already prepared; inspect existing manifest")
    data = ROOT / "2_data_storage/processed/evidence_steering_toy"
    data_manifest = json.loads((data / "manifest.json").read_text())
    snapshot = STUDY / "source_snapshot"
    config_dir = ROOT / "3_model_training/configs/evidence_steering_20261012"
    config_dir.mkdir(parents=True, exist_ok=True)
    (STUDY / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "3_model_training/emf", snapshot / "3_model_training/emf",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "4_experimental_result/control", snapshot / "control",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = snapshot / "scripts"
    scripts.mkdir()
    for name in ("steering_gpu.slurm", "steering_workflow.py", "steering_report.py"):
        source = ROOT / "3_model_training/scripts" / name
        if source.exists():
            shutil.copy2(source, scripts / name)
    review = snapshot / "review"
    review.mkdir()
    for path in (ROOT / "5_test").glob("test_steering_*.py"):
        shutil.copy2(path, review / path.name)
    stages = {}
    def config(name, variant="factorized", rank=8, seed=142, **overrides):
        values = asdict(RedesignRunConfig(
            redesign=RedesignConfig(variant=variant, rank=rank),
            data_dir=str(data), output_dir=str(STUDY / "source" / name),
            seed=seed, train_steps=20000, train_seconds=2520,
            cpu_threads=1, support_sizes=[4, 16, 64], num_attempts=256,
            nfe=32, k=16, checkpoint_every=2000, log_every=100,
            eval_seed=841203, evaluate_after_training=False))
        values.update(overrides)
        path = config_dir / (name + ".yaml")
        path.write_text(yaml.safe_dump(values, sort_keys=False))
        return {"name": name, "config": str(path)}
    operation = config("operation", train_steps=80, train_seconds=120,
                       output_dir=str(STUDY / "operation"))
    stages["operation"] = {"wall_minutes": 10, "concurrency": 1, "tasks": [operation]}
    checkpoints = [str(ROOT / f"4_experimental_result/result/calibrated_redesign_001/source/factorized_seed_{seed}/checkpoints/source_final.pt")
                   for seed in (42, 43, 44)]
    if any(not Path(p).is_file() for p in checkpoints):
        raise FileNotFoundError("original trace checkpoint missing")
    stages["trace"] = {"wall_minutes": 30, "concurrency": 1, "tasks": [{
        "arguments": ["--checkpoints", *checkpoints, "--output", str(STUDY / "trace"),
                      "--device", "cuda", "--cases-per-checkpoint", "8",
                      "--trajectories", "16", "--jacobian-cases", "8",
                      "--jacobian-trajectories", "2", "--max-seconds", "1650"]}]}
    source = []
    for seed in (142, 143):
        for name, variant, rank in (("r8", "factorized", 8), ("r32", "factorized", 32),
                                    ("conditional", "conditional", 32)):
            source.append(config(f"{name}_seed_{seed}", variant=variant, rank=rank, seed=seed))
    stages["source"] = {"wall_minutes": 45, "concurrency": 3, "tasks": source}
    stages["evaluation"] = {"wall_minutes": 10, "concurrency": 4,
                            "tasks": [{**task, "scorer": "r8_seed_" + task["name"].rsplit("_", 1)[1]}
                                      for task in source]}
    privileged = config("privileged_seed_142", output_dir=str(STUDY / "privileged/seed_142"))
    stages["privileged_train"] = {"wall_minutes": 45, "concurrency": 1, "tasks": [privileged]}
    stages["privileged_evaluation"] = {"wall_minutes": 15, "concurrency": 1,
                                      "tasks": [{**privileged, "scorer": "r8_seed_142"}]}
    pinned = [p for p in snapshot.rglob("*") if p.is_file()]
    pinned += [Path(task["config"]) for stage in stages.values()
               for task in stage["tasks"] if "config" in task]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ROOT), "study_root": str(STUDY), "snapshot": str(snapshot),
        "maximum_gpu_hours": 8.0, "maximum_concurrent_gpus": 4, "stages": stages,
        "snapshot_sha256": {str(p): sha(p) for p in sorted(set(pinned))},
        "input_sha256": {str(p): sha(p) for p in
                         [data / "manifest.json", data / "private/oracle.json",
                          ROOT / "4_experimental_result/result/exploration_001/checkpoints/flow_0004000.pt",
                          *map(Path, checkpoints)]},
        "data_manifest_sha256": sha(data / "manifest.json"),
        "family_sizes": data_manifest["family_sizes"],
        "primary_metric": "distinct unmeasured local joint-profile successes per attempted slot",
        "coefficient_access": "private evaluator only; explicitly labelled privileged diagnostic exception",
        "planned_cap_gpu_hours": sum(s["wall_minutes"] * len(s["tasks"]) / 60 for s in stages.values()),
        "model_assignments": {"data": "gpt-6-sol", "evaluation": "gpt-6-sol",
                              "scientific_review_trace": "gpt-6-astra", "reporting": "gpt-6-luna"},
    }
    if manifest["planned_cap_gpu_hours"] > 8:
        raise ValueError("planned wall limits exceed study cap")
    atomic(STUDY / "study_manifest.json", manifest)
    print(json.dumps({"manifest": str(STUDY / "study_manifest.json"),
                      "allocation_cap_gpu_hours": manifest["planned_cap_gpu_hours"]}), flush=True)


def accounting(job_ids):
    if not job_ids:
        return []
    completed = subprocess.run(
        ["sacct", "-X", "-P", "-n", "-j", ",".join(job_ids),
         "--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode"],
        text=True, capture_output=True, check=True)
    return completed.stdout.splitlines()


def wait_for(manifest, state, stages, *, required=()):
    identifiers = [state["launched"][s]["job_id"] for s in stages]
    while True:
        queue = subprocess.run(["squeue", "-h", "-j", ",".join(identifiers),
                                "-o", "%i|%j|%T|%M|%R"],
                               capture_output=True, text=True)
        if queue.returncode:
            state["scheduler_warning"] = queue.stderr
            atomic(STUDY / "workflow_state.json", state)
            time.sleep(60)
            continue
        state.update(queue=queue.stdout.splitlines(), updated_utc=datetime.now(timezone.utc).isoformat())
        atomic(STUDY / "workflow_state.json", state)
        print(json.dumps({"phase": state["phase"], "queue": state["queue"]}), flush=True)
        if not queue.stdout.strip():
            break
        time.sleep(30)
    lines = accounting(identifiers)
    state.setdefault("accounting_by_stage", {}).update({",".join(stages): lines})
    for name in required:
        job_id = state["launched"][name]["job_id"]
        stage = manifest["stages"][name]
        expected = [f"{job_id}_{i}" for i in range(len(stage["tasks"]))] if len(stage["tasks"]) > 1 else [job_id]
        records = {line.split("|")[0]: line.split("|") for line in lines}
        # Accounting sometimes appears one poll later than squeue completion.
        for _ in range(3):
            if all(identifier in records for identifier in expected):
                break
            time.sleep(5)
            records = {line.split("|")[0]: line.split("|") for line in accounting(identifiers)}
        failures = {identifier: records.get(identifier, ["missing", "MISSING"])
                    for identifier in expected
                    if identifier not in records or records[identifier][1] != "COMPLETED"
                    or records[identifier][4] != "0:0"}
        if failures:
            raise RuntimeError(f"{name} did not complete: {failures}")
    atomic(STUDY / "workflow_state.json", state)


def submit(manifest, state, stage):
    if stage in state["launched"]:
        return
    spec = manifest["stages"][stage]
    addition = spec["wall_minutes"] * len(spec["tasks"]) / 60
    committed = sum(record["allocation_cap_gpu_hours"] for record in state["launched"].values())
    if committed + addition > manifest["maximum_gpu_hours"] + 1e-9:
        raise RuntimeError("submission would exceed allocation cap")
    wall = spec["wall_minutes"]
    command = ["sbatch", "--parsable", f"--job-name=emf-steering-{stage}",
               f"--time={wall//60:02d}:{wall%60:02d}:00",
               f"--output={STUDY}/slurm_%A_%a.log"]
    if len(spec["tasks"]) > 1:
        command.append(f"--array=0-{len(spec['tasks'])-1}%{spec['concurrency']}")
    command += [str(Path(manifest["snapshot"]) / "scripts/steering_gpu.slurm"),
                str(STUDY / "study_manifest.json"), manifest["snapshot"], stage]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(result.stdout)
    state["launched"][stage] = {"job_id": job_id, "tasks": len(spec["tasks"]),
                                "allocation_cap_gpu_hours": addition,
                                "submitted_utc": datetime.now(timezone.utc).isoformat()}
    atomic(STUDY / "workflow_state.json", state)
    print(json.dumps({"event": "submitted", "stage": stage, "job_id": job_id}), flush=True)


def choose_checkpoints(manifest):
    candidates = {}
    for task in manifest["stages"]["source"]["tasks"]:
        config = yaml.safe_load(Path(task["config"]).read_text())
        output = Path(config["output_dir"])
        report = json.loads((output / "source_summary.json").read_text())
        files = {int(p.stem.split("_")[1]): str(p)
                 for p in (output / "checkpoints").glob("step_*.pt")}
        files[report["step"]] = str(output / "checkpoints/source_final.pt")
        candidates[task["name"]] = files
    common = set.intersection(*(set(values) for values in candidates.values()))
    if not common:
        raise RuntimeError("no common completed training checkpoint")
    step = max(common)
    record = {"step": step, "rule": "largest common completed update count",
              "checkpoints": {name: values[step] for name, values in candidates.items()}}
    record["sha256"] = {name: sha(path) for name, path in record["checkpoints"].items()}
    atomic(STUDY / "checkpoint_selection.json", record)
    return record


def privileged_needed():
    means = []
    for seed in (142, 143):
        path = STUDY / f"source/r8_seed_{seed}/evaluation_validation/summary.json"
        summary = json.loads(path.read_text())
        values = [r["mean_steering_benefit"] for r in summary["aggregate"]]
        means.append(sum(values) / len(values))
    return all(value <= 0 for value in means), means


def run():
    manifest_path = STUDY / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest["snapshot_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"pinned source changed: {path}")
    lock = (STUDY / "workflow.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = STUDY / "workflow_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"launched": {}}
    if state.get("phase") == "complete":
        print("Study already complete.", flush=True)
        return
    try:
        state["phase"] = "allocated GPU operation"
        submit(manifest, state, "operation")
        wait_for(manifest, state, ["operation"], required=["operation"])
        state["phase"] = "source training and existing-checkpoint trace"
        submit(manifest, state, "trace")
        submit(manifest, state, "source")
        wait_for(manifest, state, ["trace", "source"], required=["source"])
        state["checkpoint_selection"] = choose_checkpoints(manifest)
        state["phase"] = "paired evidence evaluation"
        submit(manifest, state, "evaluation")
        wait_for(manifest, state, ["evaluation"], required=["evaluation"])
        needed, means = privileged_needed()
        state["privileged_trigger"] = {"needed": needed, "validation_seed_benefits": means,
                                      "rule": "both r8 validation seed means <= 0"}
        atomic(state_path, state)
        if needed:
            state["phase"] = "privileged-context diagnostic training"
            submit(manifest, state, "privileged_train")
            wait_for(manifest, state, ["privileged_train"], required=["privileged_train"])
            state["phase"] = "privileged-context diagnostic evaluation"
            submit(manifest, state, "privileged_evaluation")
            wait_for(manifest, state, ["privileged_evaluation"], required=["privileged_evaluation"])
        state["phase"] = "complete"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        state["phase"] = "workflow stopped on error"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        ids = [record["job_id"] for record in state["launched"].values()]
        state["accounting"] = accounting(ids)
        atomic(state_path, state)
        report = Path(manifest["snapshot"]) / "scripts/steering_report.py"
        if report.exists():
            subprocess.run([sys.executable, str(report), "--study-root", str(STUDY), "--once"],
                           cwd=ROOT, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()


if __name__ == "__main__":
    main()
