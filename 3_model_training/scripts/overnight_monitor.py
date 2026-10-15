"""Refresh the overnight study dashboard and paired results in a tmux session."""

import argparse
import csv
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4_experimental_result/control"))
from study_analysis import aggregate


def dashboard(study_dir, report):
    launch_path = study_dir / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    job_ids = launch.get("all_job_ids", [])
    queue = subprocess.run(
        ["squeue", "--jobs", ",".join(job_ids), "--format=%i %j %T %M %R"]
        if job_ids else ["squeue", "--user", "miyao", "--format=%i %j %T %M %R"],
        capture_output=True, text=True, check=False)
    lines = ["EMF overnight supervision study",
             datetime.now(timezone.utc).isoformat(),
             f"Complete: {report['complete_tasks']}/18; failed: {report['failed_tasks']}",
             "Four concurrent L40S jobs; 3,600 synchronized update seconds per model.",
             "", queue.stdout.strip(), "",
             "seed  lambda  decoder       state          updates   training_seconds"]
    for task in json.loads((study_dir / "tasks.json").read_text()):
        output = Path(task["output_dir"])
        provenance = output / "task_provenance.json"
        state = json.loads(provenance.read_text()).get("state", "not_started") if provenance.exists() else "not_started"
        step, seconds = "", ""
        if (output / "training.csv").exists():
            with (output / "training.csv").open(newline="") as stream:
                last = deque(csv.DictReader(stream), maxlen=1)
            if last:
                step, seconds = last[0].get("step", ""), last[0].get("training_seconds", "")
        lines.append(f"{task['seed']:4}  {task['lambda_mid']:6g}  {task['decoder_mode']:12}  "
                     f"{state:13}  {str(step):>9}  {str(seconds):>16}")
    lines.extend(["", f"Results: {study_dir / 'analysis'}",
                  "Read paired_effect_summary.csv and quality_and_yield_vs_nfe.csv when complete."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path,
                        default=ROOT / "4_experimental_result/result/overnight_20261010")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError("Use an interval of at least ten seconds")
    while True:
        try:
            report = aggregate(study_dir=args.study_dir)
            output = dashboard(args.study_dir, report)
            (args.study_dir / "dashboard.txt").write_text(output)
            print("\033[2J\033[H" + output, flush=True)
        except Exception:
            error = traceback.format_exc()
            with (args.study_dir / "monitor_errors.log").open("a") as stream:
                stream.write(datetime.now(timezone.utc).isoformat() + "\n" + error)
            print(error, file=sys.stderr, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
