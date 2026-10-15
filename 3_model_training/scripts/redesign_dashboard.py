"""Persistent tmux display of actual workflow handoffs and Slurm allocations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

def render(root, state_path):
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {"phase": "starting", "agents": {}, "jobs": []}
    lines = ["EMF calibrated redesign — parallel agents and experiments",
             datetime.now(timezone.utc).isoformat(),
             "Inspect active collaboration agents with /agents in the Codex client.",
             f"Phase: {state.get('phase', 'unknown')}", ""]
    for role, record in state.get("agents", {}).items():
        lines.append(f"{role}: {record}")
    lines.extend(["", "Slurm jobs:"])
    queue = subprocess.run(["squeue", "-u", "miyao", "-o",
                            "%.18i %.9P %.24j %.8T %.10M %R"],
                           text=True, capture_output=True, check=False)
    lines.extend(queue.stdout.strip().splitlines())
    lines.extend(["", "Tracked job IDs: " + ", ".join(str(x) for x in state.get("jobs", []))])
    for path in sorted(root.glob("*/status.json")):
        try:
            run = json.loads(path.read_text())
            lines.append(f"{path.parent.name}: {json.dumps(run, sort_keys=True)}")
        except (OSError, ValueError):
            pass
    manifest = Path("2_data_storage/processed/calibrated_redesign_toy/manifest.json")
    if manifest.exists():
        data = json.loads(manifest.read_text())
        lines.append("Toy data ready: " + ", ".join(
            f"{name} {info['compounds']} molecules / {info['campaigns']} campaigns"
            for name, info in data["splits"].items()))
    lines.extend(["", "Daily records: 1_research_file/result_log/*_1011.md",
                  "Artifacts: " + str(root)])
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="4_experimental_result/result/calibrated_redesign_001")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        content = render(root, root / "workflow_state.json")
        (root / "dashboard.txt").write_text(content)
        print("\033[2J\033[H" + content, flush=True)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
