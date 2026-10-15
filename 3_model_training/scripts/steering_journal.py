"""Copy measured steering-study status into a marked daily research-log block.

This bridge only reads study artifacts and updates its own marked section in
``experiment_results_MMDD.md``.  It does not schedule jobs or edit study files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo


BEGIN = "<!-- EVIDENCE_STEERING_OVERNIGHT_BEGIN -->"
END = "<!-- EVIDENCE_STEERING_OVERNIGHT_END -->"
DEFAULT_STUDY_ROOT = Path("4_experimental_result/result/evidence_steering_001")
DEFAULT_LOG_DIR = Path("1_research_file/result_log")
LA = ZoneInfo("America/Los_Angeles")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value):
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _job_map(state):
    launched = state.get("launched", {})
    if isinstance(launched, dict):
        jobs = {}
        for stage, record in launched.items():
            job = record.get("job_id", record.get("jobid")) if isinstance(record, dict) else record
            if job is not None:
                jobs[str(stage)] = str(job)
        return jobs
    jobs = state.get("launched_jobids", state.get("jobs", []))
    if isinstance(jobs, dict):
        return {str(key): str(value) for key, value in jobs.items()}
    return {str(i + 1): str(value) for i, value in enumerate(jobs or [])}


def _test_run_status(analysis, expected):
    evaluations = analysis.get("evaluation_summaries", [])
    completed = []
    pending = []
    for record in evaluations:
        if record.get("split") != "test":
            continue
        summary = record.get("summary") or {}
        path = record.get("directory_key", "unknown")
        if not str(path).startswith("source/"):
            continue
        if summary.get("phase") == "evaluation_complete" and record.get("metrics_present"):
            completed.append(path)
        else:
            pending.append(path)
    completed = sorted(set(completed))
    # A test can be recorded only after both its summary and metrics exist.
    expected_text = str(expected) if expected is not None else "not recorded"
    return completed, sorted(set(pending)), expected_text


def _yield_table(analysis):
    """Mean existing seed-level yields by split, path, support and condition."""
    cells = {}
    for row in analysis.get("aggregates", []):
        split = row.get("split")
        if split not in {"validation", "test"}:
            continue
        path = row.get("directory_key")
        support = row.get("support_size")
        condition = row.get("condition")
        if not path or support is None or condition not in {"correct", "opposite"}:
            continue
        value = _number(row.get("distinct_local_profile_yield_per_attempt"))
        if value is None:
            continue
        cells.setdefault((split, str(path), str(support), condition), []).append(value)
    grouped = {}
    for (split, path, support, condition), values in cells.items():
        grouped.setdefault((split, path, support), {})[condition] = sum(values) / len(values)
    return sorted((split, path, support, values.get("correct"), values.get("opposite"))
                  for (split, path, support), values in grouped.items())


def _report_excerpt(report_path):
    try:
        text = Path(report_path).read_text(encoding="utf-8")
    except OSError:
        return "Derived report is pending."
    start_marker = "## What is being measured"
    end_marker = "## Current completion"
    start = text.find(start_marker)
    if start < 0:
        return "Derived report has not written its main measured-results section yet."
    end = text.find(end_marker, start)
    excerpt = text[start:] if end < 0 else text[start:end]
    return excerpt.strip()


def _fmt(value):
    return "pending" if value is None else f"{100 * value:.2f}%"


def _model_label(path):
    name = path.rsplit("/", 1)[-1]
    for key, label in (("r8_seed_", "8 directions"), ("r32_seed_", "32 directions"),
                       ("conditional_seed_", "Ordinary conditioning")):
        if name.startswith(key):
            return f"{label}, seed {name[len(key):]}"
    if path == "operation":
        return "Operation check (80 updates)"
    if path.startswith("privileged/"):
        return "Known campaign coefficient (diagnostic)"
    return path


def _report_block(study_root, log_dir, analysis_path, state_path, report_path, now):
    analysis = read_json(analysis_path) or {}
    state = read_json(state_path) or {}
    jobs = _job_map(state)
    manifest = read_json(Path(study_root) / "study_manifest.json") or {}
    source_launch = (state.get("launched") or {}).get("source", {})
    expected = source_launch.get("tasks") if isinstance(source_launch, dict) else None
    if expected is None:
        expected = manifest.get("source_tasks")
    completed, pending, expected_text = _test_run_status(analysis, expected)

    try:
        relative_report = os.path.relpath(report_path, start=log_dir)
    except ValueError:
        relative_report = str(report_path)

    phase = state.get("phase", analysis.get("workflow_phase", "not recorded"))
    job_text = ", ".join(f"{stage} `{job}`" for stage, job in sorted(jobs.items())) or "none recorded"
    lines = [
        BEGIN,
        "## Evidence-steering overnight status",
        "",
        f"Updated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Study phase: **{phase}**. Tracked jobs: {job_text}.",
        f"Full derived report: [analysis/report.md]({relative_report})",
        f"Completed test source runs: {len(completed)} of {expected_text} declared runs.",
        "",
        _report_excerpt(report_path),
    ]
    if completed:
        lines.append("Completed test run paths: " + ", ".join(f"`{path}`" for path in completed) + ".")
    if pending:
        lines.append("Test outputs still pending or incomplete: " + ", ".join(f"`{path}`" for path in pending) + ".")
    if not completed and not pending:
        lines.append("No completed test-model evaluation artifacts are recorded yet; missing evaluations remain pending.")
    lines.extend([
        "",
        "The intervention changes measured outcomes while keeping the model, lead, goal, and paired sampling draws fixed. `correct` uses the campaign's own measurements; `opposite` uses the paired campaign's measurement table.",
        "",
        "The response scores are simulated values from the study's declared toy response functions, not biological assay measurements. The withheld-response diagnostic uses the shared rank-8 scorer across generator comparisons; it does not measure each model's native response quality. The values below are descriptive means across recorded campaigns and seeds. They make no uncertainty claim across campaign signs.",
        "",
        "### Correct versus opposite evidence",
        "",
        "Distinct local-profile successes per attempted slot; invalid attempts count in the denominator.",
        "",
        "| Split | Model | Measured compounds | Correct evidence yield | Opposite evidence yield |",
        "|---|---|---:|---:|---:|",
    ])
    table = _yield_table(analysis)
    if table:
        for split, path, support, correct, opposite in table:
            lines.append(f"| {split} | {_model_label(path)} | {support} | {_fmt(correct)} | {_fmt(opposite)} |")
    else:
        lines.append("| pending | pending | pending | pending | pending |")
    lines.extend(["", END])
    return "\n".join(lines)


def _replace_block(existing, block):
    start = existing.find(BEGIN)
    end = existing.find(END)
    if (start < 0) != (end < 0) or (start >= 0 and end < start):
        raise ValueError("daily journal contains an incomplete evidence-steering marker block")
    if start >= 0:
        end += len(END)
        return existing[:start] + block + existing[end:]
    separator = "\n" if existing and not existing.endswith("\n") else ""
    return existing + separator + ("\n" if existing else "") + block + "\n"


def _atomic_replace(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def refresh(study_root=DEFAULT_STUDY_ROOT, log_dir=DEFAULT_LOG_DIR, *, now=None):
    study_root = Path(study_root).resolve()
    log_dir = Path(log_dir).resolve()
    now = (now or datetime.now(LA)).astimezone(LA)
    date_label = now.strftime("%m%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"experiment_results_{date_label}.md"
    lock_path = Path(str(log_path) + ".lock")
    analysis_path = study_root / "analysis" / "summary.json"
    report_path = study_root / "analysis" / "report.md"
    state_path = study_root / "workflow_state.json"
    block = _report_block(study_root, log_dir, analysis_path, state_path, report_path, now)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if log_path.exists():
                existing = log_path.read_text(encoding="utf-8")
            else:
                existing = f"# Experiment results — {date_label}\n\n"
            updated = _replace_block(existing, block)
            _atomic_replace(log_path, updated)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {"log_path": str(log_path), "phase": (read_json(state_path) or {}).get("phase", "not recorded"),
            "completed_test_runs": _test_run_status(read_json(analysis_path) or {}, None)[0],
            "job_ids": _job_map(read_json(state_path) or {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="refresh once and exit")
    mode.add_argument("--interval", type=float, default=60,
                      help="refresh period in seconds (default: 60)")
    args = parser.parse_args(argv)
    while True:
        status = refresh(args.study_root, args.log_dir)
        print(f"journal refreshed: {status['log_path']} · phase={status['phase']} · "
              f"test runs complete={len(status['completed_test_runs'])}", flush=True)
        if args.once:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
