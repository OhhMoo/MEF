"""Execute one pinned Slurm task; the coordinator owns the immutable run matrix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=["operation", "pilot", "source", "evaluation"], required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(manifest["workspace"])
    stage = manifest["stages"][args.stage]
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task = stage["tasks"][index]
    # Check pinned source bytes before importing the training package.
    import hashlib
    for relative, expected in manifest["snapshot_sha256"].items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"pinned source changed: {relative}")
    print(json.dumps({"event": "task_start", "stage": args.stage,
                      "task": task, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}), flush=True)
    if args.stage == "operation":
        subprocess.run([sys.executable, "-m", "pytest", "5_test/test_redesign_operation.py",
                        "-m", "gpu", "-q", "-o", f"pythonpath={os.environ['PYTHONPATH']}", "--basetemp",
                        f"5_test/results/redesign_gpu_{os.environ['SLURM_JOB_ID']}_tmp",
                        "--junitxml", f"5_test/results/redesign_operation_gpu_{os.environ['SLURM_JOB_ID']}.xml"],
                       cwd=root, check=True)
    command = [sys.executable, "-u", "-m", "emf.redesign_train", "--config", str(root / task["config"])]
    if task.get("checkpoint"):
        command.extend(["--evaluate-checkpoint", str(root / task["checkpoint"])])
    subprocess.run(command, cwd=root, check=True)
    print(json.dumps({"event": "task_complete", "stage": args.stage, "index": index}), flush=True)


if __name__ == "__main__":
    main()
