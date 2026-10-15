"""Refresh a bounded, read-only report for the evidence-steering study.

The reporter reads evaluator summaries and metrics only.  It writes its
derived artifacts below ``<study-root>/analysis`` and never schedules work or
changes workflow state.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time


DEFAULT_STUDY_ROOT = Path("4_experimental_result/result/evidence_steering_001")
SPLITS = ("validation", "test")


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _read_json_value(path):
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


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path, text):
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")


def _read_metrics(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def _directory_key(eval_dir, study_root):
    """Retain run-directory identity, including rank even if variant repeats."""
    relative = Path(eval_dir).relative_to(study_root)
    parts = list(relative.parts[:-1])
    # The full run path distinguishes seeds and model capacities while
    # remaining useful in a combined CSV.  Keep a readable fallback at root.
    return "/".join(parts) if parts else "."


def collect(study_root):
    study_root = Path(study_root)
    workflow = _read_json(study_root / "workflow_state.json") or {}
    collected, summaries, missing = [], [], []
    for summary_path in sorted(study_root.rglob("summary.json")):
        split = summary_path.parent.name.removeprefix("evaluation_")
        if split not in SPLITS or "analysis" in summary_path.relative_to(study_root).parts:
            continue
        summary = _read_json(summary_path)
        if summary is None:
            continue
        eval_dir = summary_path.parent
        metrics_path = eval_dir / "metrics.csv"
        summary_record = {
            "split": split,
            "directory_key": _directory_key(eval_dir, study_root),
            "summary_path": str(summary_path.relative_to(study_root)),
            "metrics_path": (str(metrics_path.relative_to(study_root))
                             if metrics_path.exists() else None),
            "summary": summary,
            "metrics_present": metrics_path.exists(),
        }
        summaries.append(summary_record)
        if not metrics_path.exists():
            missing.append(summary_record["summary_path"].replace("summary.json", "metrics.csv"))
            continue
        for source_row in _read_metrics(metrics_path):
            row = dict(source_row)
            row["split"] = split
            row["directory_key"] = summary_record["directory_key"]
            row["summary_path"] = summary_record["summary_path"]
            row["metrics_path"] = summary_record["metrics_path"]
            # Preserve numeric CSV fields as numbers for downstream analysis;
            # empty values remain empty rather than being imputed.
            for key, value in list(row.items()):
                numeric = _number(value)
                if numeric is not None:
                    row[key] = int(numeric) if numeric.is_integer() else numeric
            collected.append(row)
    return workflow, summaries, collected, missing


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def _write_csv(path, rows):
    if not rows:
        _atomic_text(path, "")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
    _atomic_text(path, stream.getvalue())


def _job_ids(workflow):
    launched = workflow.get("launched")
    if isinstance(launched, dict):
        jobs = {}
        for stage, record in launched.items():
            if isinstance(record, dict):
                job_id = record.get("job_id", record.get("jobid", record.get("slurm_job_id")))
                if job_id is not None:
                    jobs[str(stage)] = job_id
            elif record is not None:
                jobs[str(stage)] = record
        return jobs
    for key in ("launched_jobids", "launched_job_ids", "jobids", "jobs"):
        value = workflow.get(key)
        if value:
            return value
    return []


def _format_job_ids(value):
    if isinstance(value, dict):
        return ", ".join(f"{stage}: {job}" for stage, job in value.items()) or "none recorded"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(job) for job in value) or "none recorded"
    return str(value) if value else "none recorded"


def _group_key(row):
    return (str(row.get("split", "unknown")), str(row.get("directory_key", "unknown")),
            str(row.get("source_variant") or "unknown"),
            str(row.get("source_seed") or "unknown"),
            str(row.get("support_size") or "unknown"),
            str(row.get("condition") or "unknown"))


def _attempt_summary(rows):
    def total(field):
        values = [_number(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        return sum(values) if values else None

    attempts = total("attempts")
    # Distinct successes count unique generated molecules. Raw-slot successes
    # count each attempted slot; duplicates can therefore count repeatedly.
    distinct_local = total("local_profile_success_attempts")
    raw_local = total("raw_slot_local_profile_success_attempts")
    distinct_profile = total("profile_success_attempts")
    raw_profile = total("raw_slot_profile_success_attempts")
    def rate(successes):
        return successes / attempts if successes is not None and attempts else None
    return {"attempts": attempts, "distinct_local_profile_successes": distinct_local,
            "raw_slot_local_profile_successes": raw_local,
            "distinct_profile_successes": distinct_profile,
            "raw_slot_profile_successes": raw_profile,
            "distinct_local_profile_yield_per_attempt": rate(distinct_local),
            "raw_slot_local_profile_yield_per_attempt": rate(raw_local),
            "distinct_profile_yield_per_attempt": rate(distinct_profile),
            "raw_slot_profile_yield_per_attempt": rate(raw_profile)}


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    result = []
    for key, members in sorted(groups.items()):
        split, directory, variant, seed, support, condition = key
        counts = _attempt_summary(members)
        benefits = [_number(row.get("steering_benefit")) for row in members]
        benefits = [value for value in benefits if value is not None]
        result.append({"split": split, "directory_key": directory,
                       "source_variant": variant, "source_seed": seed,
                       "support_size": support, "condition": condition,
                       "campaign_rows": len(members),
                       "steering_benefit_mean": sum(benefits) / len(benefits) if benefits else None,
                       **counts})
    return result


def _main_test_rows(rows):
    """Keep ordinary test evaluations; operation and privileged runs stay out."""
    selected = []
    for row in rows:
        key = str(row.get("directory_key", "")).lower()
        if row.get("split") != "test" or "operation" in key or "privileged" in key:
            continue
        selected.append(row)
    return selected


def _curve_model_key(directory_key):
    """Collapse per-seed paths such as ``source/r8_seed142`` to ``source/r8``."""
    parts = str(directory_key).split("/")
    if parts:
        parts[-1] = re.sub(r"_seed_?\d+$", "", parts[-1])
    return "/".join(parts) or "unknown"


def _support_curve_means(rows, metric):
    """Average campaign cells within each seed, then average source seeds."""
    campaign_cells = {}
    for row in _main_test_rows(rows):
        value = _number(row.get(metric))
        size = _number(row.get("support_size"))
        condition = str(row.get("condition", ""))
        if value is None or size is None or condition not in ("correct", "opposite"):
            continue
        model = _curve_model_key(row.get("directory_key", "unknown"))
        seed = str(row.get("source_seed", "unknown"))
        family = str(row.get("family", "unknown"))
        campaign = str(row.get("campaign", "unknown"))
        key = (model, size, condition, seed, family, campaign)
        campaign_cells.setdefault(key, []).append(value)
    seed_cells = {}
    for (model, size, condition, seed, _family, _campaign), values in campaign_cells.items():
        seed_cells.setdefault((model, size, condition, seed), []).append(sum(values) / len(values))
    model_cells = {}
    for (model, size, condition, seed), values in seed_cells.items():
        model_cells.setdefault((model, size, condition), []).append(sum(values) / len(values))
    return {key: sum(values) / len(values) for key, values in model_cells.items()}


def _campaign_sign(row):
    for field in ("coefficient", "campaign_coefficient", "s"):
        value = _number(row.get(field))
        if value is not None and value != 0:
            return 1 if value > 0 else -1
    campaign = str(row.get("campaign", "")).lower()
    matches = list(re.finditer(r"(?:^|[_\-.])(plus|positive|pos|minus|negative|neg)(?=[_\-.]|$)", campaign))
    if matches:
        return 1 if matches[-1].group(1) in {"plus", "positive", "pos"} else -1
    # steering_data.py declares each sign pair as *_a (positive) and *_b
    # (negative); the suffix is part of the public campaign construction.
    suffix = re.search(r"_([ab])$", campaign)
    if suffix:
        return 1 if suffix.group(1) == "a" else -1
    return None


def _preference_reversal_means(rows):
    """Paired-campaign means for generated Δu under correct evidence."""
    pair_cells = {}
    for row in _main_test_rows(rows):
        if str(row.get("condition", "")) != "correct":
            continue
        value = _number(row.get("mean_generated_u_change"))
        size = _number(row.get("support_size"))
        sign = _campaign_sign(row)
        if value is None or size is None or sign is None:
            continue
        campaign = str(row.get("campaign", "unknown"))
        pair_id = re.sub(r"_(?:a|b|plus|minus|positive|negative|pos|neg)$", "", campaign, flags=re.I)
        key = (_curve_model_key(row.get("directory_key", "unknown")), size,
               str(row.get("source_seed", "unknown")), str(row.get("family", "unknown")),
               pair_id, sign)
        pair_cells.setdefault(key, []).append(value)
    paired = {}
    pair_keys = {(model, size, seed, family, pair)
                 for model, size, seed, family, pair, _sign in pair_cells}
    for model, size, seed, family, pair in pair_keys:
        positive = pair_cells.get((model, size, seed, family, pair, 1))
        negative = pair_cells.get((model, size, seed, family, pair, -1))
        if positive and negative:
            paired[(model, size, seed, family, pair, 1)] = positive
            paired[(model, size, seed, family, pair, -1)] = negative
    seed_cells = {}
    for (model, size, seed, _family, _pair, sign), values in paired.items():
        seed_cells.setdefault((model, size, seed, sign), []).append(sum(values) / len(values))
    model_cells = {}
    for (model, size, _seed, sign), values in seed_cells.items():
        model_cells.setdefault((model, size, sign), []).append(sum(values) / len(values))
    return {key: sum(values) / len(values) for key, values in model_cells.items()}


def _trace_rows(study_root):
    study_root = Path(study_root)
    result = []
    for path in sorted(study_root.rglob("stages.json")):
        relative = path.relative_to(study_root)
        if "analysis" in relative.parts or "trace" not in relative.parts:
            continue
        value = _read_json_value(path)
        if isinstance(value, list):
            result.extend(row for row in value if isinstance(row, dict))
    for path in sorted(study_root.rglob("stage_metrics.csv")):
        relative = path.relative_to(study_root)
        if "analysis" in relative.parts or "trace" not in relative.parts:
            continue
        result.extend(_read_metrics(path))
    return result


def _plot_preference(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = _preference_reversal_means(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {1: "-", -1: "--"}
    groups = {}
    for (model, size, sign), value in means.items():
        groups.setdefault((model, sign), []).append((size, value))
    for (model, sign), points in sorted(groups.items()):
        points.sort()
        ax.plot([point[0] for point in points], [point[1] for point in points],
                marker="o", linestyle=styles[sign],
                label=f"{model} · {'+s' if sign > 0 else '−s'} campaign")
    ax.axhline(0, color="black", linewidth=.8, alpha=.6)
    ax.set_xlabel("Support size (measured molecules)")
    ax.set_ylabel("Mean generated change in u (Δu)")
    ax.set_title("Test preference reversal under correct evidence")
    ax.grid(True, alpha=.25)
    if groups:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(.5, .5, "Pending: no test correct-evidence Δu measurements with identifiable campaign signs",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
    fig.tight_layout()
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    _atomic_bytes(path, buffer.getvalue())
    return len(means)


def _plot_trace(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("correction_l2", "Composed correction ΔX"),
               ("logit_delta_l2", "Decoder logit change"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    plotted = 0
    for ax, (field, title) in zip(axes, metrics):
        groups = {}
        for row in rows:
            value = _number(row.get(field))
            probe = _number(row.get("time"))
            comparison = row.get("comparison")
            if value is None or value <= 0 or probe is None or not comparison:
                continue
            groups.setdefault(str(comparison), {}).setdefault(probe, []).append(value)
        for comparison, by_time in sorted(groups.items()):
            times = sorted(by_time)
            values = [sum(by_time[t]) / len(by_time[t]) for t in times]
            ax.plot(times, values, marker="o", label=comparison)
            plotted += len(times)
        ax.set_yscale("log")
        ax.set_xlabel("Flow time t")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=.25)
        if groups:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(.5, .5, "Pending trace stage measurements", ha="center", va="center",
                    transform=ax.transAxes)
    axes[0].set_ylabel("L2 change (log scale; zero values omitted)")
    fig.suptitle("Evidence-route trace across flow time")
    fig.tight_layout()
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    _atomic_bytes(path, buffer.getvalue())
    return plotted


def _plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    curve_fields = (
        ("local_profile_yield_per_attempt", "Distinct successful molecules"),
        ("raw_slot_local_profile_yield_per_attempt", "Raw successful slots"),
    )
    styles = {"correct": "-", "opposite": "--"}
    plotted = False
    selected = _main_test_rows(rows)
    for ax, (field, title) in zip(axes, curve_fields):
        means = _support_curve_means(selected, field)
        groups = {}
        for (model, size, condition), value in means.items():
            groups.setdefault((model, condition), []).append((size, value))
        for (model, condition), points in sorted(groups.items()):
            points.sort()
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    marker="o", linestyle=styles[condition], label=f"{model} · {condition}")
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("Support size (measured molecules)")
        ax.grid(True, alpha=.25)
        if groups:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(.5, .5, "No main test measurements yet", ha="center", va="center",
                    transform=ax.transAxes)
    axes[0].set_ylabel("Local profile yield per attempted slot")
    fig.suptitle("Test evidence steering · solid: correct · dashed: opposite")
    fig.tight_layout()
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    _atomic_bytes(path, buffer.getvalue())


def _report(workflow, summaries, rows, missing, study_root):
    phase = workflow.get("phase", "not recorded")
    launched = _format_job_ids(_job_ids(workflow))
    lines = [
        "# Evidence-steering study status",
        "",
        f"Refreshed: {datetime.now(timezone.utc).isoformat()}",
        f"Study directory: `{study_root}`",
        "",
        "## What is being measured",
        "",
        "This study changes the measured outcomes while holding the model, lead, goal, and paired sampling draws fixed. `correct` uses the campaign's own measurements; `opposite` replays the paired campaign's proposals under the other campaign's measurements. A positive steering benefit means correct evidence produced a higher local profile yield per attempted slot than opposite evidence.",
        "",
        "Yields use every attempted slot as the denominator, including invalid and duplicate proposals. Distinct yields count each successful molecule once; raw-slot yields count a repeated successful molecule in every slot where it appeared. The support curves show local profile yield, with separate labels for distinct and raw-slot counting.",
        "",
        "## Current completion",
        "",
        f"Workflow phase: **{phase}**. Launched job IDs recorded in workflow state: {launched}.",
        f"Evaluation summaries found: {len(summaries)}; metric rows read: {len(rows)}.",
    ]
    if missing:
        lines.append("Metrics files pending beside these summaries: " + ", ".join(f"`{p}`" for p in missing) + ".")
    if not summaries:
        lines.append("No validation or test summary has been written yet. Missing measurements remain pending; they are not treated as zero.")
    elif not rows:
        lines.append("Summaries exist, but no evaluation metric rows are available yet. No yields or successes are inferred from the summaries alone.")
    lines.extend(["", "## Measured comparisons", ""])
    if not rows:
        lines.append("No measured comparisons are available yet.")
    else:
        lines.append("Each line pools campaign rows within one split, run directory, source seed, support size, and condition. Success counts and rates retain their denominators.")
        lines.append("")
        lines.append("| Split / run / model / seed / support / condition | Attempts | Distinct local successes / yield | Raw-slot local successes / yield | Mean steering benefit |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in aggregate(rows):
            title = (f"{item['split']} / {item['directory_key']} / {item['source_variant']} / "
                     f"{item['source_seed']} / N={item['support_size']} / {item['condition']}")
            def cell(success_key, yield_key):
                value = item[yield_key]
                rate = "pending" if value is None else f"{value:.4f}"
                count = item[success_key]
                count_text = "pending" if count is None else f"{count:g}"
                return f"{count_text} / {rate}"
            benefit = item["steering_benefit_mean"]
            benefit_text = "pending" if benefit is None else f"{benefit:+.4f}"
            attempts = "pending" if item["attempts"] is None else f"{item['attempts']:g}"
            lines.append(f"| `{title}` | {attempts} | {cell('distinct_local_profile_successes', 'distinct_local_profile_yield_per_attempt')} | {cell('raw_slot_local_profile_successes', 'raw_slot_local_profile_yield_per_attempt')} | {benefit_text} |")
    lines.extend(["", "## Source artifacts", ""])
    if summaries:
        for item in summaries:
            lines.append(f"- `{item['summary_path']}` ({item['split']}, run `{item['directory_key']}`); metrics: `{item['metrics_path'] or 'pending'}`")
    else:
        lines.append("- No evaluation summaries found.")
    lines.extend(["", "Figures: `support_curves.png` shows distinct and raw-slot local yields; `preference_reversal.png` compares generated Δu for paired positive and negative campaigns; `evidence_trace.png` shows the composed correction and decoder-logit changes across flow time. Campaign signs use a row coefficient when present, otherwise the last explicit plus/minus token; the study's declared `_a`/`_b` suffix convention marks positive/negative campaigns. Only complete sign pairs contribute. The preference and trace figures show pending placeholders when their measurements are absent.", "", "Detailed rows are in `combined_metrics.csv`. The numbers are simulated measurements from the study's declared response functions, not biological assay results.", ""])
    return "\n".join(lines)


def refresh(study_root):
    study_root = Path(study_root)
    analysis = study_root / "analysis"
    workflow, summaries, rows, missing = collect(study_root)
    aggregates = aggregate(rows)
    trace_rows = _trace_rows(study_root)
    preference_points = _preference_reversal_means(rows)
    trace_points = sum(1 for row in trace_rows
                       if _number(row.get("time")) is not None
                       and (_number(row.get("correction_l2")) is not None
                            or _number(row.get("logit_delta_l2")) is not None))
    _atomic_json(analysis / "summary.json", {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_root": str(study_root),
        "workflow_phase": workflow.get("phase"),
        "launched_jobids": _job_ids(workflow),
        "evaluation_summaries": summaries,
        "missing_metrics_paths": missing,
        "metric_rows": len(rows),
        "preference_reversal_points": len(preference_points),
        "trace_stage_rows": len(trace_rows),
        "aggregates": aggregates,
        "observations_are_simulated": True,
    })
    _write_csv(analysis / "combined_metrics.csv", rows)
    _plot(analysis / "support_curves.png", rows)
    _plot_preference(analysis / "preference_reversal.png", rows)
    _plot_trace(analysis / "evidence_trace.png", trace_rows)
    _atomic_text(analysis / "report.md", _report(workflow, summaries, rows, missing, study_root) + "\n")
    return {"phase": workflow.get("phase", "not recorded"),
            "summaries": len(summaries), "metric_rows": len(rows),
            "missing_metrics": len(missing), "preference_reversal_points": len(preference_points),
            "trace_stage_rows": len(trace_rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="refresh once and exit")
    mode.add_argument("--interval", type=float, default=60,
                      help="refresh period in seconds (default: 60)")
    args = parser.parse_args(argv)
    while True:
        status = refresh(args.study_root)
        print(f"steering report refreshed: phase={status['phase']}; "
              f"summaries={status['summaries']}; rows={status['metric_rows']}; "
              f"pending_metrics={status['missing_metrics']}", flush=True)
        if args.once:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
