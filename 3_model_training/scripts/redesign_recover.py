"""Bounded recovery of interrupted evaluations without changing trained models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(".recovery.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def execute(manifest, stage):
    root = Path(manifest["workspace"])
    for name, expected in manifest["snapshot_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"pinned recovery source changed: {name}")
    if stage == "probe":
        subprocess.run([sys.executable, "-m", "pytest", "5_test/test_redesign_operation.py",
                        "-m", "gpu", "-q", "-o", f"pythonpath={os.environ['PYTHONPATH']}",
                        "--basetemp", f"5_test/results/redesign_recovery_gpu_{os.environ['SLURM_JOB_ID']}_tmp",
                        "--junitxml", f"5_test/results/redesign_recovery_gpu_{os.environ['SLURM_JOB_ID']}.xml"],
                       cwd=root, check=True)
        tasks = manifest["probe_tasks"]
    else:
        tasks = [manifest["tasks"][int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    for task in tasks:
        print(json.dumps({"event":"recovery_task_start", "stage":stage, "task":task}), flush=True)
        subprocess.run([sys.executable, "-u", "-m", "emf.redesign_train", "--config", str(root / task["config"]),
                        "--evaluate-checkpoint", str(root / task["checkpoint"])], cwd=root, check=True)
        print(json.dumps({"event":"recovery_task_complete", "name":task["name"]}), flush=True)


def launch(manifest_path, manifest):
    root = Path(manifest["workspace"])
    study = Path(manifest["study_root"])
    state_path = study / "workflow_state.json"
    state = json.loads(state_path.read_text())
    recovery = state.get("recovery", {})
    if manifest["allocation_cap_gpu_hours"] + manifest["prior_actual_gpu_hours"] > manifest["maximum_gpu_hours"]:
        raise RuntimeError("recovery would exceed original allocated-GPU budget")
    for stage in ["probe", "evaluation"]:
        if stage in recovery:
            continue
        command = ["sbatch", "--parsable", "--cpus-per-task=1", f"--job-name=emf-recover-{stage}",
                   "--time=00:10:00" if stage == "probe" else "--time=00:15:00"]
        if stage == "probe":
            command.append("--partition=debug")
        else:
            command.extend([f"--array=0-{len(manifest['tasks'])-1}%4",
                            f"--dependency=afterok:{recovery['probe']['job_id']}", "--kill-on-invalid-dep=yes"])
        command.extend(["3_model_training/scripts/redesign_recovery.slurm", str(manifest_path), stage])
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected scheduler response: {result.stdout}")
        recovery[stage] = {"job_id":job_id, "tasks":1 if stage=="probe" else len(manifest["tasks"]),
                           "submitted_utc":datetime.now(timezone.utc).isoformat()}
        state = json.loads(state_path.read_text())
        state.update(recovery=recovery, phase="recovery submitted", recovery_manifest=str(manifest_path),
                     recovery_allocation_cap_gpu_hours=manifest["allocation_cap_gpu_hours"])
        state["jobs"] = list(dict.fromkeys(state.get("jobs", []) + [job_id]))
        atomic_json(state_path, state)
        print(json.dumps({"stage":stage, **recovery[stage]}), flush=True)
    identifiers = ",".join(value["job_id"] for value in recovery.values())
    while True:
        result = subprocess.run(["squeue", "-h", "-j", identifiers, "-o", "%i|%T|%M|%R"],
                                text=True, capture_output=True)
        state = json.loads(state_path.read_text())
        state.update(phase="recovery running or queued", queue=result.stdout.splitlines(),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        if result.returncode:
            state["recovery_scheduler_error"] = result.stderr.strip()
        if not result.stdout.strip() and result.returncode == 0:
            accounting = subprocess.run(["sacct", "-j", identifiers, "-X", "-P", "-n",
                                         "--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode"],
                                        text=True, capture_output=True)
            state["recovery_accounting"] = accounting.stdout.splitlines()
            state["phase"] = "recovery scheduler finished; inspect evaluation exit states"
            atomic_json(state_path, state)
            print(accounting.stdout, flush=True)
            break
        atomic_json(state_path, state)
        print(result.stdout, flush=True)
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "execute"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", choices=["probe", "evaluation"])
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if args.command == "launch":
        launch(manifest_path, manifest)
    elif args.stage:
        execute(manifest, args.stage)
    else:
        parser.error("execute requires --stage")


if __name__ == "__main__":
    main()
