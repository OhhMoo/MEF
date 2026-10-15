"""Build a live, plain-language report for the masked-edit study.

This script only reads study inputs and writes derived files under the study
root. It is safe to rerun while tasks are still writing their outputs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile


DEFAULT_STUDY = Path("4_experimental_result/result/masked_control_001")


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def atomic_text(path, content):
    path = Path(path)
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


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def atomic_csv(path, rows):
    if not rows:
        atomic_text(path, "")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)
    atomic_text(path, stream.getvalue())


def rel(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def load_tasks(root):
    tasks = []
    for path in sorted((root / "tasks").rglob("*.json")) if (root / "tasks").exists() else []:
        task = read_json(path)
        if task:
            tasks.append({"task": task, "task_path": rel(path, root)})
    return tasks


def explicit_status(root):
    found = []
    for path in sorted(root.rglob("status.json")):
        if "analysis" in path.relative_to(root).parts:
            continue
        value = read_json(path)
        if value is not None:
            found.append({"path": rel(path, root), **value})
            for container_key in ("tasks", "stages", "jobs", "launched"):
                container = value.get(container_key)
                if isinstance(container, dict):
                    entries = container.items()
                elif isinstance(container, list):
                    entries = ((str(i), item) for i, item in enumerate(container))
                else:
                    continue
                for identity, record in entries:
                    if isinstance(record, dict):
                        found.append({"path": rel(path, root), **record,
                                      "task": record.get("task", record.get("name", identity))})
                    elif record is not None:
                        found.append({"path": rel(path, root), "task": identity,
                                      "job_id": record})
    return found


def status_for(task, task_path, statuses, output):
    # Prefer a status file inside the task output, then a matching task/name
    # status record elsewhere in the study.
    output_path = Path(output).resolve() if output else None
    for record in statuses:
        path = (Path(record["path"]).parent).resolve()
        if output_path is not None and (path == output_path or output_path in path.parents):
            return record
    for record in statuses:
        identity = str(record.get("task", record.get("name", record.get("task_name", ""))))
        if identity and identity in {str(task.get("name", "")), Path(task_path).stem,
                                     str(task.get("stage", ""))}:
            return record
    return {}


def normalize_state(raw, artifacts):
    state = str(raw or "").strip().lower()
    if state in {"complete", "completed", "success", "succeeded", "done"}:
        return "completed"
    if state in {"failed", "error", "cancelled", "canceled", "timeout"}:
        return "failed"
    if state in {"running", "started", "active", "in_progress", "in-progress"}:
        return "running"
    if artifacts:
        return "completed"
    return "pending"


def collect(root):
    root = Path(root).resolve()
    manifest = read_json(root / "study_manifest.json") or {}
    root_status = read_json(root / "status.json") or {}
    statuses = explicit_status(root)
    tasks = load_tasks(root)
    stages = []
    for item in tasks:
        task, task_path = item["task"], item["task_path"]
        output_value = task.get("output")
        output = Path(output_value) if output_value else root
        if not output.is_absolute():
            output = root / output
        training_path = output / "training.csv"
        training_summary_path = output / "training_summary.json"
        validation_dir = output / "validation"
        summaries = sorted(output.rglob("summary.json")) if output.exists() else []
        status_record = status_for(task, task_path, statuses, output_value)
        artifact_done = training_summary_path.exists() or any(
            p.exists() for p in (validation_dir / "summary.json", output / "summary.json"))
        raw_status = (status_record.get("status", status_record.get("state", task.get("status")))
                      or root_status.get("status_by_task", {}).get(task.get("name")))
        stage = str(task.get("stage", "unknown"))
        if raw_status is None and not artifact_done and training_path.exists():
            raw_status = "running"
        state = normalize_state(raw_status, artifact_done)
        if state == "pending" and output.exists() and any(output.iterdir()):
            state = "running"
        training_summary = read_json(training_summary_path) or {}
        validation_summary = read_json(validation_dir / "summary.json") or {}
        timestamp = (status_record.get("updated_utc") or status_record.get("finished_utc")
                     or status_record.get("started_utc") or status_record.get("timestamp")
                     or validation_summary.get("completed_utc") or training_summary.get("finished_utc"))
        job_id = (status_record.get("job_id") or status_record.get("jobid")
                  or status_record.get("slurm_job_id") or task.get("job_id"))
        stages.append({
            "name": task.get("name", Path(task_path).stem), "stage": stage,
            "variant": task.get("variant", "unknown"), "seed": task.get("seed", "unknown"),
            "state": state, "raw_status": raw_status, "job_id": job_id,
            "timestamp": timestamp or "unknown", "task_path": task_path,
            "output": rel(output, root), "steps_requested": task.get("steps"),
            "training_path": rel(training_path, root) if training_path.exists() else None,
            "training_summary_path": rel(training_summary_path, root) if training_summary_path.exists() else None,
            "summary_paths": [rel(p, root) for p in summaries],
        })

    # Keep discovered, non-operation control/revision summaries visible even
    # when their workflow did not write a task JSON.
    known = {(s["output"], s["stage"], str(s["variant"]), str(s["seed"])) for s in stages}
    summary_records = []
    for path in sorted(root.rglob("*.json")):
        # Read only primary result summaries: including live_report.json here
        # recursively embeds each previous refresh in the next report.
        if path.name not in {"summary.json", "training_summary.json", "control_summary.json", "fewshot_summary.json", "revision_summary.json", "result_summary.json"}:
            continue
        if any(part.startswith("source_snapshot") for part in path.relative_to(root).parts):
            continue
        if "tasks" in path.relative_to(root).parts or "analysis" in path.relative_to(root).parts:
            continue
        value = read_json(path)
        if not value or path.name in {"study_manifest.json", "status.json"}:
            continue
        followup_name = path.name in {"control_summary.json", "fewshot_summary.json",
                                      "revision_summary.json", "result_summary.json"}
        followup_path = any(token in "/".join(path.parts).lower()
                            for token in ("control", "fewshot", "few_shot", "revision"))
        if not followup_name and not followup_path and not any(
                k in value for k in ("phase", "stage", "variant", "seed", "per_mask", "protocol", "calibration")):
            continue
        record = {"path": rel(path, root), "value": value}
        summary_records.append(record)

    aggregate_rows, attempts_rows, training_rows = [], [], []
    for stage in stages:
        output = root / stage["output"]
        training_rows.extend({"task": stage["name"], "stage": stage["stage"],
                              "variant": stage["variant"], "seed": stage["seed"],
                              "training_path": stage["training_path"], **row}
                             for row in read_csv(output / "training.csv"))
        summary_paths = [output / "validation" / "summary.json", output / "summary.json"]
        for summary_path in summary_paths:
            summary = read_json(summary_path)
            if not summary:
                continue
            base = {"record_type": "validation_per_mask", "task": stage["name"], "stage": stage["stage"],
                    "variant": summary.get("variant", stage["variant"]),
                    "seed": summary.get("seed", stage["seed"]),
                    "phase": summary.get("phase", "unknown"),
                    "summary_path": rel(summary_path, root),
                    "completed_utc": summary.get("completed_utc", summary.get("finished_utc", "unknown")),
                    "exact_target_reconstruction": summary.get("exact_target_reconstruction"),
                    "logical_attempts": summary.get("logical_attempts"),
                    "physical_attempts": summary.get("physical_attempts"),
                    "nfe_per_sample": summary.get("nfe_per_sample"),
                    "seconds_including_chemistry": summary.get("seconds_including_chemistry")}
            aggregate_rows.append({**base, "record_type": "reconstruction",
                                   "mask_id": "all", "exact_target_reconstruction":
                                   summary.get("exact_target_reconstruction"),
                                   "attempts": None, "valid_fraction": None,
                                   "protected_fraction": None, "distinct_eligible": None,
                                   "distinct_eligible_yield": None, "mean_graph_edits": None})
            per_mask = summary.get("per_mask", {})
            if isinstance(per_mask, dict):
                for mask_id, values in sorted(per_mask.items()):
                    if isinstance(values, dict):
                        aggregate_rows.append({**base, "record_type": "validation_per_mask",
                            "exact_target_reconstruction": None, "mask_id": mask_id,
                            "attempts": values.get("attempts"),
                            "valid_fraction": values.get("valid_fraction"),
                            "protected_fraction": values.get("protected_fraction"),
                            "distinct_eligible": values.get("distinct_eligible"),
                            "distinct_eligible_yield": values.get("distinct_eligible_yield"),
                            "mean_graph_edits": values.get("mean_graph_edits")})
            attempts_path = summary_path.parent / "attempts.csv"
            for row in read_csv(attempts_path):
                attempts_rows.append({"task": stage["name"], "stage": stage["stage"],
                    "variant": base["variant"], "seed": base["seed"],
                    "attempts_path": rel(attempts_path, root), **row})

    # Capture summaries for later control, few-shot, and revision stages even
    # when their output schema or task-manifest convention differs.
    task_outputs = {str(Path(stage["output"])) for stage in stages}
    followup_summaries = []
    for record in summary_records:
        path = Path(record["path"])
        value = record["value"]
        if path.name not in {"summary.json", "control_summary.json", "fewshot_summary.json",
                             "revision_summary.json", "result_summary.json"}:
            continue
        parent = Path(*path.parts[:-1])
        if any(parent == Path(output) or Path(output) in parent.parents for output in task_outputs):
            continue
        # Training summaries are represented in the task's operation section.
        if path.name == "training_summary.json":
            continue
        stage_name = str(value.get("stage", "/".join(path.parts[:-1]) or "unknown"))
        followup = {"record_type": "followup_summary", "stage": stage_name,
                    "summary_path": record["path"]}
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                followup[key] = item
            elif isinstance(item, (dict, list)):
                followup[key] = item
        followup_summaries.append(followup)
        aggregate_rows.append(followup)

    return {"study_root": str(root), "generated_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest, "root_status": root_status, "stages": stages,
            "status_files": statuses, "summaries": summary_records,
            "followup_summaries": followup_summaries,
            "aggregate_rows": aggregate_rows, "attempt_rows": attempts_rows,
            "training_rows": training_rows}


def fmt(value, digits=3):
    n = number(value)
    return "unknown" if n is None else f"{n:.{digits}f}"


def plain_summary(data):
    stages = data["stages"]
    summary_by_stage = {}
    for record in data["summaries"]:
        value = record["value"]
        key = (str(value.get("stage", "")), str(value.get("variant", "")), str(value.get("seed", "")))
        summary_by_stage[key] = value
    lines = ["# Masked-edit study: live report", "",
             f"Report refreshed: {data['generated_utc']}", "",
             "This study first compares three ways to represent the same local graph edit: "
             "A anchors an editable residual to the lead, B predicts editable graph coordinates "
             "from a zero-centered start, and C combines the anchored start with masked prediction. "
             "B and C can describe the same endpoints; this comparison measures how their training "
             "and sampling packages behave.", "",
             "The planned follow-up is terminal-response control, few-shot calibration, and trajectory "
             "revision. Exact known-response control uses privileged oracle information and is reported "
             "as a separate upper-bound diagnostic. A calibrated controller uses a learned molecular "
             "surrogate and estimates response coefficients from simulated support observations in the "
             "known feature basis; this is a simulation study, not wet-lab calibration.", "",
             "## Work status", ""]
    if stages:
        for item in stages:
            suffix = f"; Slurm job {item['job_id']}" if item.get("job_id") is not None else ""
            when = item.get("timestamp", "unknown")
            lines.append(f"- **{item['name']}** ({item['stage']}, variant {item['variant']}, seed {item['seed']}): "
                         f"{item['state']}; timestamp {when}{suffix}.")
    else:
        lines.append("No task manifests are present yet; stage status is unknown.")
    lines += ["", "## A/B/C operation results", "",
              "Validation measures are computed over attempted samples, including invalid attempts in the denominator. "
              "Validity means chemically valid and connected. Protected fraction measures exact protected-category "
              "preservation. Distinct eligible counts unique valid, connected, protected non-lead outputs. "
              "Exact target reconstruction is a separate decoder reconstruction check and is not generation yield.", ""]
    ops = [s for s in stages if s["stage"] == "operation"]
    if not ops:
        lines.append("Operation results are pending.")
    for stage in ops:
        s = read_json(Path(data["study_root"]) / stage["output"] / "validation" / "summary.json") or {}
        if not s:
            lines.append(f"- Variant {stage['variant']}, seed {stage['seed']}: results pending ({stage['state']}).")
            continue
        lines.append(f"- **Variant {stage['variant']}, seed {stage['seed']}**: "
                     f"{s.get('logical_attempts', 'unknown')} logical attempts; exact target reconstruction "
                     f"{fmt(s.get('exact_target_reconstruction'))}; {s.get('nfe_per_sample', 'unknown')} flow evaluations "
                     "per sample. Per-mask results:")
        per_mask = s.get("per_mask", {})
        if not per_mask:
            lines.append("  - Per-mask measurements are pending.")
        for mask_id, m in sorted(per_mask.items()):
            lines.append(f"  - {mask_id}: valid fraction {fmt(m.get('valid_fraction'))}; protected fraction "
                         f"{fmt(m.get('protected_fraction'))}; {m.get('distinct_eligible', 'unknown')} distinct "
                         f"eligible molecules from {m.get('attempts', 'unknown')} attempts "
                         f"(yield {fmt(m.get('distinct_eligible_yield'))}).")
    lines += ["", "## Control, few-shot, and revision results", ""]
    followups = data.get("followup_summaries", [])
    if not followups:
        lines.append("No follow-up summary files are present yet; these results are pending.")
    for item in followups:
        name = item.get("stage", "unknown stage")
        context = item.get("context_mode", item.get("control_mode", item.get("controller", "unknown")))
        phase = item.get("phase", item.get("status", "unknown"))
        lines.append(f"- **{name}** (context/control: {context}; phase/status: {phase}), "
                     f"from `{item['summary_path']}`.")
        shown = []
        for key in ("attempts", "logical_attempts", "successes", "distinct_successes",
                    "success_rate", "valid_fraction", "protected_fraction", "nfe",
                    "elapsed_seconds", "seconds_including_chemistry", "completed_utc"):
            value = item.get(key)
            if value is not None:
                shown.append(f"{key.replace('_', ' ')} {value}")
        if shown:
            lines.append("  - Recorded: " + "; ".join(shown) + ".")
    if data["status_files"]:
        lines += ["", "## Recorded status details", ""]
        for status in data["status_files"]:
            lines.append(f"- `{status['path']}`: status {status.get('status', status.get('state', 'unknown'))}; "
                         f"job ID {status.get('job_id', status.get('jobid', status.get('slurm_job_id', 'unknown')))}; "
                         f"time {status.get('updated_utc', status.get('finished_utc', status.get('timestamp', 'unknown')))}.")
    lines += ["", "## Artifacts", "",
              "Machine-readable details are in `live_report.json`, with flattened `masked_edit_aggregate.csv`, "
              "`masked_edit_attempts.csv`, and `masked_edit_training.csv` beside this report.", ""]
    return "\n".join(lines)


def refresh(study):
    study = Path(study).resolve()
    data = collect(study)
    atomic_json(study / "live_report.json", data)
    atomic_csv(study / "masked_edit_aggregate.csv", data["aggregate_rows"])
    atomic_csv(study / "masked_edit_attempts.csv", data["attempt_rows"])
    atomic_csv(study / "masked_edit_training.csv", data["training_rows"])
    atomic_text(study / "live_report.md", plain_summary(data))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY,
                        help=f"study root (default: {DEFAULT_STUDY})")
    args = parser.parse_args()
    result = refresh(args.study)
    print(f"Refreshed report for {len(result['stages'])} task(s) under {Path(args.study).resolve()}")


if __name__ == "__main__":
    main()
