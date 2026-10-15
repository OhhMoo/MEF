"""Report the local-edit overnight study and maintain its daily log block."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo


DEFAULT_STUDY_ROOT = Path("4_experimental_result/result/local_edit_001")
DEFAULT_LOG_DIR = Path("1_research_file/result_log")
BEGIN = "<!-- LOCAL_EDIT_OVERNIGHT_BEGIN -->"
END = "<!-- LOCAL_EDIT_OVERNIGHT_END -->"
LA = ZoneInfo("America/Los_Angeles")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _to_number(value):
    number = _number(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return value


def _read_csv(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return [{key: _to_number(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)]
    except (OSError, csv.Error):
        return []


def _atomic_text(path, content):
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
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")


def _flatten_metadata(value):
    result = {}
    cursor = value
    for _ in range(8):
        if not isinstance(cursor, dict):
            break
        result.update({key: item for key, item in cursor.items() if key != "metadata"})
        cursor = cursor.get("metadata")
    return result


def _job_ids(state):
    launched = state.get("launched", {})
    if isinstance(launched, dict):
        return {str(stage): str(record.get("job_id", record.get("jobid")))
                for stage, record in launched.items()
                if isinstance(record, dict) and record.get("job_id", record.get("jobid")) is not None}
    value = state.get("launched_jobids", state.get("jobs", []))
    return value if isinstance(value, (dict, list)) else []


def _evaluation_info(metrics_path, study_root):
    eval_dir = Path(metrics_path).parent
    basename = eval_dir.name
    split = "test" if basename.endswith("_test") or basename == "evaluation_test" else (
        "validation" if basename.endswith("_validation") or basename == "evaluation_validation" else None)
    if split is None:
        return None
    relative = eval_dir.relative_to(study_root)
    parts = list(relative.parts)
    if parts and parts[-1] == basename:
        parts.pop()
    run_label = "/".join(parts) or "."
    control_radius = None
    if basename.startswith("control_"):
        control_radius = basename.removeprefix("control_").removesuffix(f"_{split}")
        run_label = f"{run_label}/control_{control_radius}" if run_label != "." else f"control_{control_radius}"
    return eval_dir, split, run_label, control_radius


def _ancestor_json(eval_dir, study_root, filename):
    for directory in (eval_dir, *eval_dir.parents):
        if directory == study_root.parent:
            break
        value = _read_json(directory / filename)
        if isinstance(value, dict):
            return value
    return {}


def _context_description(protocol, task, control_radius):
    extension = protocol.get("extension", {}) if isinstance(protocol, dict) else {}
    mode = extension.get("context_mode") or task.get("context_mode")
    kind = extension.get("kind") or task.get("kind")
    if mode in {"known", "privileged", "true_coefficient"}:
        description = "true coefficient (privileged diagnostic)"
    elif mode == "bridge":
        description = "measured evidence → frozen H → source-trained scalar head"
    elif mode in {"inferred", "evidence", "measured"}:
        description = "measured response evidence"
    elif mode == "calibrated_control":
        description = "measured evidence → support-LS scalar + source-fitted surrogate"
    else:
        description = "context mode pending"
    if control_radius is not None:
        description += f"; control radius {control_radius}"
    return {"context_mode": mode, "context_description": description, "task_kind": kind}


def _generation_eligibility(eval_dir):
    values = {}
    for path in sorted(eval_dir.rglob("generation_summary.json")):
        if "analysis" in path.relative_to(eval_dir).parts:
            continue
        summary = _flatten_metadata(_read_json(path) or {})
        campaign = summary.get("campaign")
        support = summary.get("support_size")
        condition = summary.get("condition")
        eligible = _number(summary.get("eligible_distinct"))
        if campaign is None or support is None or condition is None or eligible is None:
            continue
        key = (str(campaign), str(support), str(condition))
        values.setdefault(key, []).append(eligible)
    return {key: sum(counts) for key, counts in values.items()}


def collect(study_root):
    study_root = Path(study_root).resolve()
    manifest = _read_json(study_root / "study_manifest.json") or {}
    state = _read_json(study_root / "workflow_state.json") or {}
    decision = _read_json(study_root / "continuation_decision.json")
    statuses = []
    for path in sorted(study_root.rglob("status.json")):
        if "analysis" in path.relative_to(study_root).parts:
            continue
        status = _read_json(path)
        if status is not None:
            statuses.append({"path": str(path.relative_to(study_root)), **status})

    rows, eval_paths, pending_metrics = [], [], []
    for metrics_path in sorted(study_root.rglob("metrics.csv")):
        if "analysis" in metrics_path.relative_to(study_root).parts:
            continue
        info = _evaluation_info(metrics_path, study_root)
        if info is None:
            continue
        eval_dir, split, run_label, control_radius = info
        protocol = _read_json(eval_dir / "protocol.json") or {}
        task = _ancestor_json(eval_dir, study_root, "local_edit_task.json")
        context = _context_description(protocol, task, control_radius)
        eligibility = _generation_eligibility(eval_dir)
        eval_paths.append(str(eval_dir.relative_to(study_root)))
        records = _read_csv(metrics_path)
        if not records:
            pending_metrics.append(str(metrics_path.relative_to(study_root)))
        for record in records:
            row = dict(record)
            row.update(split=split, source_path=run_label,
                       metrics_path=str(metrics_path.relative_to(study_root)),
                       **context)
            campaign = row.get("campaign")
            support = row.get("support_size")
            condition = row.get("condition")
            key = (str(campaign), str(support), str(condition))
            if _number(row.get("eligible_distinct_attempts")) is None and key in eligibility:
                row["eligible_distinct_attempts"] = eligibility[key]
            if _number(row.get("nfe")) is None:
                row["nfe"] = protocol.get("nfe")
            rows.append(row)

    # A protocol with no metrics file is still an explicitly pending evaluation.
    for path in sorted(study_root.rglob("protocol.json")):
        if "analysis" in path.relative_to(study_root).parts:
            continue
        info = _evaluation_info(path.parent / "metrics.csv", study_root)
        if info is None:
            continue
        relative_metrics = str((path.parent / "metrics.csv").relative_to(study_root))
        if not (path.parent / "metrics.csv").exists():
            pending_metrics.append(relative_metrics)
            eval_paths.append(str(path.parent.relative_to(study_root)))

    return {"study_root": str(study_root), "manifest": manifest, "workflow_state": state,
            "continuation_decision": decision, "statuses": statuses,
            "evaluation_paths": sorted(set(eval_paths)),
            "pending_metrics_paths": sorted(set(pending_metrics)), "rows": rows}


def _sum_present(rows, field):
    values = [_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _sum_aliases(rows, *fields):
    for field in fields:
        value = _sum_present(rows, field)
        if value is not None:
            return value
    return None


def _mean_present(rows, field):
    values = [_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows):
    groups = {}
    keys = ("source_path", "split", "source_seed", "family", "support_size",
            "condition", "context_description", "task_kind")
    for row in rows:
        key = tuple(str(row.get(name) if row.get(name) is not None else "pending") for name in keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, members in sorted(groups.items()):
        item = dict(zip(keys, key))
        attempts = _sum_present(members, "attempts")
        valid = _sum_present(members, "connected_valid_attempts")
        profile_hits = _sum_aliases(members, "profile_success_attempts",
                                    "distinct_profile_success_attempts")
        local_hits = _sum_aliases(members, "local_profile_success_attempts",
                                  "distinct_local_profile_success_attempts")
        eligible = _sum_present(members, "eligible_distinct_attempts")
        paired_attempts = _sum_present(members, "paired_decoded_changed_attempts")
        paired_fractions = [_number(row.get("paired_decoded_changed_fraction")) for row in members]
        paired_fractions = [value for value in paired_fractions if value is not None]
        nfe_values = sorted({str(value) for row in members
                             if (value := _number(row.get("nfe"))) is not None})
        item.update({
            "campaign_rows": len(members), "attempts": attempts,
            "distinct_valid_profile_hits": profile_hits,
            "distinct_local_profile_hits": local_hits,
            "eligible_distinct": eligible,
            "connected_valid_attempts": valid,
            "connected_valid_rate": valid / attempts if valid is not None and attempts else None,
            "profile_yield_per_attempt": profile_hits / attempts if profile_hits is not None and attempts else None,
            "local_profile_yield_per_attempt": local_hits / attempts if local_hits is not None and attempts else None,
            "paired_decoded_changed_attempts": paired_attempts,
            "paired_decoded_changed_fraction_mean": (sum(paired_fractions) / len(paired_fractions)
                                                     if paired_fractions else None),
            "nfe": ", ".join(nfe_values) if nfe_values else None,
        })
        result.append(item)
    return result


def _fmt(value, digits=3):
    if value is None:
        return "pending"
    return f"{value:.{digits}f}"


def _display(value):
    return "pending" if value is None else str(value)


def _phase_final(phase):
    value = str(phase or "").lower()
    return value.startswith("complete") or "failed" in value or "stopped on error" in value


def _decision_paragraphs(decision, manifest):
    """Summarize selection and recovery gates without copying the decision JSON."""
    if not isinstance(decision, dict):
        return (
            "Validation-only continuation selection is pending; no winner or gate result has been recorded.",
            "Recovery selection is pending; no recovery winner or recovery gate result has been recorded.",
        )

    candidates = decision.get("candidates", [])
    candidate_by_name = {
        str(item.get("name")): item
        for item in candidates if isinstance(item, dict) and item.get("name") is not None
    }
    winner = decision.get("winner")

    def gate_details(assessment):
        if not isinstance(assessment, dict):
            return "gate details pending"
        details = []
        if assessment.get("reason"):
            details.append(str(assessment["reason"]))
        checks = assessment.get("checks", [])
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                if check.get("signed_u_reversal") is False:
                    details.append(f"{check.get('family', 'family')} signed-U reversal failed")
                elif check.get("signed_u_reversal") is True:
                    details.append(f"{check.get('family', 'family')} signed-U reversal passed")
                for key, label in (("local_eligible_yield", "eligible local yield"),
                                   ("valid", "valid-slot fraction")):
                    value = _number(check.get(key))
                    if value is not None:
                        details.append(f"{check.get('seed', 'pooled')}/{check.get('family', 'family')} {label} {value:.3f}")
        return "; ".join(details[:3]) or ("all recorded gates passed" if assessment.get("passed") else "gate details pending")

    if winner is None:
        reason = decision.get("reason") or "no candidate is recorded as passing"
        failed = []
        for name, candidate in candidate_by_name.items():
            assessment = candidate.get("assessment", {})
            if isinstance(assessment, dict) and not assessment.get("passed", False):
                failed.append(f"{name}: {gate_details(assessment)}")
        gate_text = "; ".join(failed[:3]) or "candidate gate details are pending"
        first = (f"Validation-only selection has no winner ({reason}). Gate review: {gate_text}.")
    else:
        selected = candidate_by_name.get(str(winner), {})
        assessment = selected.get("assessment", {}) if isinstance(selected, dict) else {}
        gate_result = gate_details(assessment)
        first = (f"Validation-only selection chose `{winner}`; its recorded locality, validity, and steering gates {gate_result}. "
                 f"Decision reason: {decision.get('reason') or 'not recorded'}.")

    recovery = decision.get("recovery_winner")
    recovery_candidates = decision.get("recovery_candidates", [])
    recovery_details = []
    for item in recovery_candidates:
        if not isinstance(item, dict):
            continue
        assessment = item.get("assessment", {})
        if isinstance(assessment, dict) and not assessment.get("passed", False):
            label = item.get("weight", "unnamed")
            detail = gate_details(assessment)
            recovery_details.append(f"weight {label}: {detail}")
    if recovery is None:
        if recovery_candidates:
            why = "; ".join(recovery_details[:3]) or "no recovery candidate passed the validation gates"
            second = f"Recovery selection has no winner. Gate review: {why}."
        elif decision.get("winner") is not None:
            second = "Recovery was not needed because the primary validation winner passed its gates."
        else:
            second = "Recovery selection and its validation gates are pending."
    else:
        second = (f"Recovery selection chose terminal supervision weight `{recovery}` after its validation gates passed; "
                  "the resulting calibration continuation is recorded separately from the primary selection.")
    return first, second


def _budget_note(manifest):
    manifest = manifest if isinstance(manifest, dict) else {}
    previous = _number(manifest.get("previous_study_gpu_hours"))
    current = _number(manifest.get("maximum_gpu_hours"))
    combined = _number(manifest.get("combined_ceiling_gpu_hours"))
    if previous is None or current is None or combined is None:
        return "GPU-hour budget provenance is pending in the study manifest."
    return (f"Budget accounting: the previous attempt was allocated/used {previous:.3f} GPU-hours; this study is capped at "
            f"{current:.0f} GPU-hours, for a combined ceiling of {combined:.3f} GPU-hours.")


def _historical_attempt_note(state):
    history = state.get("historical_attempts") if isinstance(state, dict) else None
    if not history:
        return None
    if isinstance(history, dict):
        attempts = [{"attempt": key, **(value if isinstance(value, dict) else {"status": value})}
                    for key, value in history.items()]
    elif isinstance(history, list):
        attempts = [item for item in history if isinstance(item, dict)]
    else:
        attempts = []
    if not attempts:
        return "Historical attempt/recovery records are present; they describe execution history and are separate from scientific outcome measurements."
    descriptions = []
    for item in attempts[:3]:
        label = item.get("attempt", item.get("name", item.get("id", "prior attempt")))
        phase = item.get("phase", item.get("status", "status not recorded"))
        job = item.get("job_id")
        detail = item.get("error", item.get("reason", item.get("message")))
        fragment = f"{label}: {phase}"
        if job is not None:
            fragment += f" (job {job})"
        if detail:
            detail = str(detail).strip().replace("\n", " ")
            fragment += f" — {detail[:240]}{'…' if len(detail) > 240 else ''}"
        descriptions.append(fragment)
    more = len(attempts) - len(descriptions)
    suffix = f"; and {more} additional record(s)" if more else ""
    return ("Historical operations: " + "; ".join(descriptions) + suffix +
            ". Setup or operation failures are execution history, not scientific outcome measurements; the recovery run is assessed separately.")


def _status_rows(statuses):
    return [{"path": row["path"], "phase": row.get("phase", "not recorded"),
             "step": row.get("step"), "training_seconds": row.get("training_seconds"),
             "wall_seconds": row.get("wall_seconds"), "error": row.get("error")}
            for row in statuses]


def build_report(data, aggregates, now):
    state = data["workflow_state"]
    phase = state.get("phase", "not recorded")
    jobs = state.get("launched", {})
    job_text = ", ".join(f"{name} `{record.get('job_id', 'pending')}`"
                          for name, record in sorted(jobs.items())
                          if isinstance(record, dict)) or "none recorded"
    decision = data.get("continuation_decision")
    lines = [
        "# Local-edit overnight study report", "",
        f"Updated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Workflow phase: **{phase}**. Tracked jobs: {job_text}.",
        f"Study manifest: `{data['study_root']}/study_manifest.json`.",
        "",
        "This study tests whether a model can make a small, chemically local change to familiar chemistry and reverse the preferred direction when measured outcomes reverse. The chemistry source is familiar; the scalar coefficient is held out. All response scores are from the study's simulated response functions.",
        "",
        "Context labels come from the task record and evaluation protocol. `true coefficient (privileged diagnostic)` uses the known hidden coefficient. `bridge` means measured support outcomes passed through frozen H and a source-trained scalar head; `calibrated_control` means measured support outcomes passed through a support least-squares scalar and a source-fitted surrogate. These evidence-based conditions are distinct from the privileged diagnostic and are not pooled.",
        "",
        _budget_note(data.get("manifest")),
        "",
        "## Workflow and training status", "",
    ]
    historical = _historical_attempt_note(state)
    if historical:
        lines[lines.index("## Workflow and training status") - 1:lines.index("## Workflow and training status")] = [historical, ""]
    status_rows = _status_rows(data["statuses"])
    if status_rows:
        lines.extend(["| Run status path | Phase | Step | Training seconds | Wall seconds |",
                      "|---|---|---:|---:|---:|"])
        for row in status_rows:
            lines.append(f"| `{row['path']}` | {row['phase']} | {_display(row['step'])} | {_display(row['training_seconds'])} | {_display(row['wall_seconds'])} |")
    else:
        lines.append("No run status files have been written yet; training status is pending.")
    stages = (data.get("manifest") or {}).get("stages", {})
    if stages:
        planned = []
        for name, spec in sorted(stages.items()):
            tasks = spec.get("tasks", []) if isinstance(spec, dict) else []
            planned.append(f"{name}: {len(tasks)} task{'s' if len(tasks) != 1 else ''}")
        lines.extend(["", "Manifest plan: " + "; ".join(planned) + "."])
    if decision is not None:
        lines.extend(["", "Continuation decision: `continuation_decision.json` is present."])
        lines.extend(_decision_paragraphs(decision, data.get("manifest")))
    else:
        lines.extend(["", "Continuation decision: pending.", *_decision_paragraphs(None, data.get("manifest"))])
    lines.extend(["", "## Paired evaluation measurements", "",
                  "Each row pools campaign records within one run, split, source seed, family, support size, condition, and context. Attempts remain the denominator, so invalid proposals count as failures. Distinct profile hits and distinct local-profile hits are reported separately. Eligible counts are included only when recorded in metrics or generation summaries.", "",
                  "| Run / condition | Split | Seed | Family | N | Context | Attempts | Connected valid / rate | Eligible distinct | Distinct profile hits | Distinct local-profile hits | Paired SMILES changed | NFE |",
                  "|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    if aggregates:
        for row in aggregates:
            valid = (f"{_display(row['connected_valid_attempts'])} / {_fmt(row['connected_valid_rate'])}"
                     if row["connected_valid_attempts"] is not None and row["connected_valid_rate"] is not None
                     else "pending")
            changed = row["paired_decoded_changed_attempts"]
            fraction = row["paired_decoded_changed_fraction_mean"]
            changed_text = (f"{_display(changed)} / {_fmt(fraction)}" if changed is not None or fraction is not None
                            else "pending")
            label = row["source_path"]
            if row["task_kind"] != "pending":
                label = f"{label} ({row['task_kind']})"
            if row["condition"] != "pending":
                label += f" · {row['condition']}"
            lines.append(f"| `{label}` | {row['split']} | {row['source_seed']} | {row['family']} | {row['support_size']} | {row['context_description']} | {_display(row['attempts'])} | {valid} | {_display(row['eligible_distinct'])} | {_display(row['distinct_valid_profile_hits'])} | {_display(row['distinct_local_profile_hits'])} | {changed_text} | {_display(row['nfe'])} |")
    else:
        lines.append("| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |")
    if data["pending_metrics_paths"]:
        lines.extend(["", "Evaluation outputs still pending metrics:"])
        lines.extend(f"- `{path}`" for path in data["pending_metrics_paths"])
    lines.extend(["", "## Paired correct/opposite summary", "",
                  "This compact view compares the local-profile yield for matched correct and opposite evidence at each run, split, seed, family, and support size. It shows descriptive paired measurements; it makes no uncertainty claim across conditions.", "",
                  "| Run | Split | Seed | Family | N | Correct local yield / attempts | Opposite local yield / attempts |",
                  "|---|---|---:|---|---:|---:|---:|"])
    pairs = {}
    for row in aggregates:
        key = tuple(row.get(field) for field in ("source_path", "split", "source_seed", "family", "support_size", "context_description"))
        pairs.setdefault(key, {})[row.get("condition")] = row
    pair_count = 0
    for key, values in sorted(pairs.items(), key=lambda item: tuple(str(v) for v in item[0])):
        if not ({"correct", "opposite"} & set(values)):
            continue
        run, split, seed, family, support, context = key
        correct = values.get("correct")
        opposite = values.get("opposite")
        def yield_attempt(item):
            if item is None:
                return "pending"
            rate, attempts = item.get("local_profile_yield_per_attempt"), item.get("attempts")
            if rate is None or attempts is None:
                return "pending"
            return f"{rate:.4f} / {attempts:g}"
        lines.append(f"| `{run}` ({context}) | {split} | {seed} | {family} | {support} | {yield_attempt(correct)} | {yield_attempt(opposite)} |")
        pair_count += 1
    if not pair_count:
        lines.append("| pending | pending | pending | pending | pending | pending | pending |")
    lines.extend(["", "Values summarize simulated measurements and generated proposals. A zero is shown only when the completed metric row measured zero successes; missing results are labeled pending."])
    return "\n".join(lines) + "\n"


def _replace_marked_block(existing, block):
    start = existing.find(BEGIN)
    end = existing.find(END)
    if (start < 0) != (end < 0) or (start >= 0 and end < start):
        raise ValueError("daily log contains an incomplete local-edit marker block")
    if start >= 0:
        end += len(END)
        return existing[:start] + block + existing[end:]
    separator = "\n" if existing and not existing.endswith("\n") else ""
    return existing + separator + ("\n" if existing else "") + block + "\n"


def _journal_block(report_path, summary, now, log_dir):
    relative_report = os.path.relpath(report_path, start=Path(log_dir).resolve())
    state = summary["workflow_state"]
    jobs = state.get("launched", {})
    job_text = ", ".join(f"{stage} `{record.get('job_id', 'pending')}`"
                          for stage, record in sorted(jobs.items())
                          if isinstance(record, dict)) or "none recorded"
    lines = [BEGIN, "## Local-edit overnight update", "",
             f"Updated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
             f"Workflow phase: **{state.get('phase', 'not recorded')}**. Tracked jobs: {job_text}.",
             f"Full report: [analysis/report.md]({relative_report})",
             "The report distinguishes the privileged true-coefficient diagnostic from measured evidence: bridge runs use frozen H plus a source-trained scalar head, while calibrated controls use a support least-squares scalar plus a source-fitted surrogate. The chemistry is familiar and the coefficient is held out; scores are simulated and attempted slots are the denominator.",
             _budget_note(summary.get("study_manifest_provenance")),
             *([_historical_attempt_note(state)] if _historical_attempt_note(state) else []),
             "", "Continuation decision", "",
             *summary.get("continuation_paragraphs", _decision_paragraphs(None, {})),
             "", "| Run / condition | Split | Seed | Family | N | Attempts | Distinct profile hits | Distinct local-profile hits | Local yield / attempt |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    rows = summary.get("aggregates", [])
    if rows:
        for row in rows:
            lines.append(f"| `{row['source_path']}` · {row['condition']} | {row['split']} | {row['source_seed']} | {row['family']} | {row['support_size']} | {_display(row['attempts'])} | {_display(row['distinct_valid_profile_hits'])} | {_display(row['distinct_local_profile_hits'])} | {_fmt(row['local_profile_yield_per_attempt'], 4)} |")
    else:
        lines.append("| pending | pending | pending | pending | pending | pending | pending | pending | pending |")
    lines.extend(["", END])
    return "\n".join(lines)


def _update_journal(log_dir, report_path, summary, now):
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    date_label = now.strftime("%m%d")
    path = log_dir / f"experiment_results_{date_label}.md"
    lock_path = Path(str(path) + ".lock")
    block = _journal_block(report_path, summary, now, log_dir)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                existing = path.read_text(encoding="utf-8")
            else:
                existing = f"# Experiment results — {date_label}\n\n"
            _atomic_text(path, _replace_marked_block(existing, block))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return path


def refresh(study_root=DEFAULT_STUDY_ROOT, log_dir=DEFAULT_LOG_DIR, *, now=None):
    study_root = Path(study_root).resolve()
    now = (now or datetime.now(LA)).astimezone(LA)
    data = collect(study_root)
    aggregates = aggregate(data["rows"])
    status = data["workflow_state"].get("phase", "not recorded")
    analysis = study_root / "analysis"
    summary = {
        "updated_at_local": now.isoformat(), "study_root": str(study_root),
        "study_manifest": {
            "created_utc": data["manifest"].get("created_utc"),
            "planned_stage_tasks": {
                name: len(spec.get("tasks", []))
                for name, spec in data["manifest"].get("stages", {}).items()
                if isinstance(spec, dict)
            },
        },
        "study_manifest_provenance": {
            key: data["manifest"].get(key) for key in
            ("previous_study_gpu_hours", "maximum_gpu_hours", "combined_ceiling_gpu_hours")
            if data["manifest"].get(key) is not None
        },
        "workflow_phase": status, "workflow_state": data["workflow_state"],
        "job_ids": _job_ids(data["workflow_state"]),
        "continuation_summary": {
            "winner": data["continuation_decision"].get("winner"),
            "recovery_winner": data["continuation_decision"].get("recovery_winner"),
            "reason": data["continuation_decision"].get("reason"),
        } if isinstance(data["continuation_decision"], dict) else None,
        "continuation_paragraphs": list(_decision_paragraphs(data["continuation_decision"], data["manifest"])),
        "source_statuses": data["statuses"],
        "evaluation_paths": data["evaluation_paths"],
        "pending_metrics_paths": data["pending_metrics_paths"],
        "metric_rows": len(data["rows"]), "aggregates": aggregates,
        "context_note": "true coefficients are privileged diagnostics; measured response evidence is a separate condition",
        "scores_are_simulated": True,
    }
    _atomic_json(analysis / "summary.json", summary)
    report = build_report(data, aggregates, now)
    _atomic_text(analysis / "report.md", report)
    journal = _update_journal(log_dir, analysis / "report.md", summary, now)
    return {"phase": status, "metric_rows": len(data["rows"]),
            "groups": len(aggregates), "pending_metrics": len(data["pending_metrics_paths"]),
            "report_path": str(analysis / "report.md"), "journal_path": str(journal)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--watch", action="store_true", help="refresh until study completes or fails")
    parser.add_argument("--interval", type=float, default=60,
                        help="poll interval in seconds (default: 60)")
    args = parser.parse_args(argv)
    while True:
        result = refresh(args.study_root, args.log_dir)
        print(f"local-edit report refreshed: phase={result['phase']}; "
              f"metric rows={result['metric_rows']}; pending={result['pending_metrics']}",
              flush=True)
        if not args.watch or _phase_final(result["phase"]):
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
