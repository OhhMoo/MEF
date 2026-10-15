"""Execute one immutable task from the constrained revision study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from emf.masked_edit_run import execute, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    task = json.loads(Path(args.task).read_text())
    if isinstance(task, list):
        task = task[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    setup(int(task["seed"]), args.device)
    if task["stage"] == "revision_efficiency":
        from emf.masked_edit_revision_benchmark import run
        run(task, args.device)
    else:
        execute(task, args.device)


if __name__ == "__main__":
    main()
