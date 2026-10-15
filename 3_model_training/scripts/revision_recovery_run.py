"""Execute a frozen task from the final revision-recovery study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from emf.masked_edit_run import setup, now
from emf.redesign_train import atomic_json


def task_digest(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    data = json.loads(Path(args.task).read_text())
    task = data[int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))] if isinstance(data, list) else data
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    setup(int(task["seed"]), args.device)
    atomic_json(output / "run_started.json", {
        "task": task, "task_sha256": task_digest(task), "started_utc": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })
    try:
        if task["stage"] == "adapter_train":
            from emf.revision_adapter import train
            result = train(task, args.device)
        else:
            from emf.revision_recovery import run
            result = run(task, args.device)
        if result.get("phase") != "complete":
            raise RuntimeError("Task returned without a complete result")
        atomic_json(output / "task_complete.json", {
            "phase": "complete", "task": task, "task_sha256": task_digest(task),
            "completed_utc": now(),
        })
    except Exception as error:
        atomic_json(output / "task_failed.json", {
            "task": task, "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "failed_utc": now(),
        })
        raise


if __name__ == "__main__":
    main()
