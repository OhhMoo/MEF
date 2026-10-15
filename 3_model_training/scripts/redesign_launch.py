"""Launch the bounded, dependency-ordered multi-agent study and keep its status live."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--interval", type=float, default=20.)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["workspace"])
    study = Path(manifest["study_root"])
    state_path = study / "workflow_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    launched = state.setdefault("launched", {})
    budget = sum(stage["wall_minutes"] / 60 * len(stage["tasks"])
                 for stage in manifest["stages"].values())
    if budget > manifest["maximum_gpu_hours"]:
        raise RuntimeError(f"allocation caps exceed GPU budget: {budget}")
    dependency = None
    for name in ["operation", "pilot", "source", "evaluation"]:
        stage = manifest["stages"][name]
        if name not in launched:
            command = ["sbatch", "--parsable", f"--job-name=emf-redesign-{name}",
                       f"--time=00:{stage['wall_minutes']:02d}:00",
                       f"--cpus-per-task={manifest.get('cpus_per_gpu_task',8)}"]
            if name in {"operation", "pilot"}:
                command.append("--partition=debug")
            if len(stage["tasks"]) > 1:
                command.append(f"--array=0-{len(stage['tasks'])-1}%{manifest['maximum_concurrent_gpus']}")
            if dependency:
                command.extend([f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"])
            command.extend(["3_model_training/scripts/redesign_gpu.slurm", str(manifest_path), name])
            result = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"unexpected scheduler response: {result.stdout!r}")
            launched[name] = {"job_id": job_id, "tasks": len(stage["tasks"]),
                              "wall_minutes": stage["wall_minutes"],
                              "dependency": dependency, "submitted_utc": datetime.now(timezone.utc).isoformat()}
            print(json.dumps({"event": "submitted", "stage": name, **launched[name]}), flush=True)
            state.update(phase=f"{name} submitted", launched=launched,
                         allocation_cap_gpu_hours=budget,
                         jobs=list(dict.fromkeys(state.get("jobs", []) + [job_id])))
            atomic_json(state_path, state)
        dependency = launched[name]["job_id"]
    job_ids = ",".join(value["job_id"] for value in launched.values())
    while True:
        queue = subprocess.run(["squeue", "-h", "-j", job_ids,
                                "-o", "%i|%j|%T|%M|%R"], text=True, capture_output=True)
        state = json.loads(state_path.read_text())
        state.update(phase="monitoring submitted study", queue=queue.stdout.splitlines(),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        if queue.returncode:
            state["scheduler_error"] = queue.stderr.strip()
        atomic_json(state_path, state)
        print(queue.stdout or "No study jobs currently queued; checking accounting.", flush=True)
        if not queue.stdout.strip() and queue.returncode == 0:
            accounting = subprocess.run(["sacct", "-j", job_ids, "-X", "-P", "-n",
                                         "--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode"],
                                        text=True, capture_output=True)
            state["accounting"] = accounting.stdout.splitlines()
            state["phase"] = "scheduler study finished; inspect task exit states and result summaries"
            atomic_json(state_path, state)
            print(accounting.stdout, flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
