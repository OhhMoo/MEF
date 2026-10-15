"""Poll calibrated-redesign artifacts and maintain measured daily summaries.

Writes only the study's analysis directory and the marked regions of the two
daily journals. Scheduling and GPU submission belong to the coordinator.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np


BEGIN = "<!-- EMF_REDESIGN_REPORT_BEGIN -->"
END = "<!-- EMF_REDESIGN_REPORT_END -->"
SUPPORT_SIZES = (16, 32, 64, 128)
METRICS = (
    "success", "efficiency",
    "pre_ranking_unmeasured_joint_profile_yield_per_attempt",
    "pre_ranking_neighbourhood_joint_profile_yield_per_attempt",
    "joint_event_brier_score", "generated_pool_response_mean_log_likelihood",
)
MODE_NAMES = {
    "native": "Native evidence",
    "swapped": "Swapped flow evidence",
    "base_only": "Selector disabled",
    "standard_proposals": "Standard proposals",
    "task_embedding": "Fitted task state",
    "selector_adaptation": "Adapted selector",
    "predictor_search": "Predictor search",
}
JOB_PATTERN = re.compile(r"^\d+(?:_\d+)?$")
ACCOUNTING_PATTERN = re.compile(r"^\d+(?:_\d+|_\[[0-9,%\-]+\])?$")


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows:
        atomic_text(path, "")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, prefix=path.name + ".",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list))
                         else value for key, value in row.items()} for row in rows)
    temporary.replace(path)


def flatten_metadata(value):
    """Merge nested generation/commit metadata, retaining the native provenance."""
    merged = {}
    cursor = value
    for _ in range(8):
        if not isinstance(cursor, dict):
            break
        merged.update({key: item for key, item in cursor.items() if key != "metadata"})
        cursor = cursor.get("metadata")
    return merged


def run_stage(path, study_root):
    parts = Path(path).relative_to(study_root).parts
    if any("operation" in part.lower() or part.lower().startswith("op_") for part in parts):
        return "operation"
    if any("pilot" in part.lower() for part in parts):
        return "pilot"
    return "controlled"


def _protocol(path, study_root, cache):
    for parent in path.parents:
        if parent == study_root.parent:
            break
        protocol_path = parent / "protocol.json"
        if protocol_path not in cache:
            cache[protocol_path] = read_json(protocol_path)
        if cache[protocol_path]:
            return cache[protocol_path]
    return {}


def collect_artifacts(study_root):
    """Summaries alone supply metrics; predictions never substitute for truth."""
    study_root = Path(study_root)
    cases, generations, sources, history, ignored = [], [], [], [], []
    protocol_cache = {}
    for path in sorted(study_root.rglob("evaluation_summary.json")):
        if "analysis" in path.relative_to(study_root).parts:
            continue
        summary = read_json(path)
        if not summary:
            continue
        metadata = flatten_metadata(summary)
        protocol = _protocol(path, study_root, protocol_cache)
        generation = read_json(path.parent / "generation_summary.json") or {}
        row = dict(metadata)
        row.update(
            artifact=str(path.relative_to(study_root)),
            stage=run_stage(path, study_root),
            architecture=metadata.get("source_variant", "unknown"),
            seed=metadata.get("source_seed"),
            campaign=metadata.get("campaign", "unknown"),
            lead=metadata.get("lead_id", metadata.get("lead_index", "unknown")),
            mode=metadata.get("mode", path.parent.name),
            eval_split=protocol.get("eval_split", metadata.get("eval_split", "unknown")),
            nfe=metadata.get("nfe", generation.get("nfe", protocol.get("nfe"))),
            support_size=summary.get("support_size", metadata.get("support_size")),
            attempts=summary.get("attempts"),
            k=summary.get("allocated_selection_slots", protocol.get("k")),
            outcome_source=summary.get("evaluation_kind"),
            generation_seconds=generation.get("generation_seconds"),
            response_scoring_seconds=generation.get("response_scoring_seconds"),
            sampling_plan_seconds=generation.get("sampling_plan_seconds"),
            end_to_end_proposal_seconds=generation.get("end_to_end_proposal_seconds"),
            connected_valid_attempts=generation.get("connected_valid_attempts"),
            supported_connected_valid_attempts=generation.get("supported_connected_valid_attempts"),
            eligible_distinct=generation.get("eligible_distinct"),
        )
        # Values are external-evaluation observables, not predicted profile scores.
        if row["outcome_source"] != "exact_simulator":
            ignored.append({"artifact": row["artifact"], "reason": "different outcome source",
                            "outcome_source": row["outcome_source"]})
            continue
        row["full_protocol"] = (
            number(row["attempts"]) == 1000 and number(row["k"]) == 16
            and number(row["support_size"]) in SUPPORT_SIZES
            and (number(row["nfe"]) == 32
                 or row["mode"] == "predictor_search" and number(row["nfe"]) in (0, None))
        )
        if row["stage"] == "controlled" and not row["full_protocol"]:
            row["stage"] = "exploratory_other"
        cases.append(row)
    for path in sorted(study_root.rglob("generation_summary.json")):
        if "analysis" in path.relative_to(study_root).parts:
            continue
        summary = read_json(path)
        if summary:
            row = flatten_metadata(summary)
            row.update(artifact=str(path.relative_to(study_root)),
                       stage=run_stage(path, study_root))
            generations.append(row)
    for manifest_path in sorted(study_root.rglob("manifest.json")):
        if "analysis" in manifest_path.relative_to(study_root).parts:
            continue
        manifest = read_json(manifest_path) or {}
        config = manifest.get("config", {})
        architecture = config.get("redesign", {}).get("variant")
        if architecture is None:
            continue
        directory = manifest_path.parent
        status = read_json(directory / "source_summary.json") or read_json(directory / "status.json") or {}
        metrics = status.get("metrics", {})
        row = {
            "run": str(directory.relative_to(study_root)),
            "stage": run_stage(directory, study_root),
            "architecture": architecture, "seed": config.get("seed"),
            "slurm_job_id": manifest.get("slurm_job_id"),
            "phase": status.get("phase", "manifest_ready"),
            "step": status.get("step"),
            "parameter_count": manifest.get("parameter_count"),
            "trainable_parameters": manifest.get("trainable_parameters"),
            "training_seconds": status.get("training_seconds"),
            "feature_cache_seconds": status.get("feature_cache_seconds"),
            "wall_seconds": status.get("wall_seconds"),
            "peak_torch_memory_gib": (number(status.get("peak_torch_memory_bytes"), 0) / 2**30
                                      if status.get("peak_torch_memory_bytes") is not None else None),
            "representation_sha256": manifest.get("representation_sha256"),
            "frozen_representation_unchanged": status.get("frozen_representation_unchanged"),
            **{key: metrics.get(key) for key in ("loss", "flow_loss", "response_loss", "decode_loss", "accepted_fraction")},
        }
        sources.append(row)
        try:
            with (directory / "training.csv").open() as stream:
                training_rows = list(csv.DictReader(stream))
            for sample in training_rows:
                if number(sample.get("step")) is not None:
                    history.append({"run": row["run"], "stage": row["stage"],
                                    "architecture": architecture, "seed": row["seed"],
                                    **{key: number(value) for key, value in sample.items()}})
        except (OSError, csv.Error):
            pass
    return cases, generations, sources, history, ignored


def clustered_estimate(rows, metric, *, bootstrap=2000, seed=271828):
    """Average leads, then campaigns/source seeds; bootstrap those units."""
    cells = defaultdict(list)
    for row in rows:
        value = number(row.get(metric))
        if value is not None:
            cells[(str(row.get("seed")), str(row.get("campaign")))].append(value)
    if not cells:
        return {"mean": None, "ci_low": None, "ci_high": None, "campaigns": 0, "seeds": 0, "cells": 0}
    seeds = sorted({key[0] for key in cells})
    campaigns = sorted({key[1] for key in cells})
    matrix = np.full((len(campaigns), len(seeds)), np.nan)
    for i, campaign in enumerate(campaigns):
        for j, source_seed in enumerate(seeds):
            values = cells.get((source_seed, campaign))
            if values:
                matrix[i, j] = np.mean(values)
    mean = float(np.nanmean(np.nanmean(matrix, axis=1)))
    low, high = None, None
    if len(campaigns) >= 2:
        rng = np.random.default_rng(seed)
        campaign_draws = rng.integers(len(campaigns), size=(bootstrap, len(campaigns)))
        seed_draws = rng.integers(len(seeds), size=(bootstrap, len(seeds)))
        samples = matrix[campaign_draws[:, :, None], seed_draws[:, None, :]]
        seed_counts = np.isfinite(samples).sum(axis=2)
        campaign_means = np.divide(np.nansum(samples, axis=2), seed_counts,
                                   out=np.full(seed_counts.shape, np.nan),
                                   where=seed_counts > 0)
        campaign_counts = np.isfinite(campaign_means).sum(axis=1)
        means = np.nansum(campaign_means, axis=1)[campaign_counts > 0] / campaign_counts[campaign_counts > 0]
        low, high = (float(value) for value in np.quantile(means, [0.025, 0.975]))
    return {"mean": mean, "ci_low": low, "ci_high": high, "campaigns": len(campaigns),
            "seeds": len(seeds), "cells": len(cells),
            "complete_seed_campaign_grid": len(cells) == len(seeds) * len(campaigns)}


def aggregate_curves(cases):
    groups = defaultdict(list)
    for row in cases:
        groups[(row["stage"], row["eval_split"], row["architecture"],
                row["mode"], row["support_size"])].append(row)
    curves = []
    for key, rows in sorted(groups.items(), key=lambda item: (
            *(str(x) for x in item[0][:-1]), number(item[0][-1], 0))):
        curve = dict(zip(("stage", "eval_split", "architecture", "mode", "support_size"), key))
        curve["cases"] = len(rows)
        for metric in METRICS:
            estimate = clustered_estimate(rows, metric)
            curve[metric] = estimate["mean"]
            curve[metric + "_ci_low"] = estimate["ci_low"]
            curve[metric + "_ci_high"] = estimate["ci_high"]
            if metric == "success":
                curve.update({name: estimate[name] for name in
                              ("campaigns", "seeds", "cells", "complete_seed_campaign_grid")})
        curve["full_pool_outcome_reader_calls"] = sum(number(row.get("full_pool_outcome_reader_calls"), 0) for row in rows)
        curve["actual_selected_endpoint_measurements"] = sum(number(row.get("actual_selected_endpoint_measurements"), 0) for row in rows)
        curve["generation_seconds"] = sum(number(row.get("generation_seconds"), 0) for row in rows)
        curves.append(curve)
    return curves


def paired_differences(cases):
    """All comparisons match seed/campaign/lead/N and the common scorer hash."""
    controls = [row for row in cases if row["stage"] == "controlled"]
    native = {}
    for row in controls:
        if row["architecture"] == "factorized" and row["mode"] == "native":
            key = (row["eval_split"], row["seed"], row["campaign"], row["lead"], row["support_size"])
            native[key] = row
    differences, mismatches = [], []
    for row in controls:
        if row["architecture"] == "factorized" and row["mode"] == "native":
            continue
        key = (row["eval_split"], row["seed"], row["campaign"], row["lead"], row["support_size"])
        reference = native.get(key)
        if reference is None:
            continue
        common = row.get("common_scorer_checkpoint_sha256")
        if not common or common != reference.get("common_scorer_checkpoint_sha256"):
            mismatches.append({"artifact": row["artifact"], "reason": "common scorer differs"})
            continue
        pair = {
            "eval_split": row["eval_split"], "seed": row["seed"], "campaign": row["campaign"],
            "lead": row["lead"], "support_size": row["support_size"],
            "comparison": f"{row['architecture']}:{row['mode']} minus factorized:native",
            "common_scorer_checkpoint_sha256": common,
            **{metric: number(row.get(metric)) - number(reference.get(metric))
               for metric in METRICS if number(row.get(metric)) is not None
               and number(reference.get(metric)) is not None},
        }
        differences.append(pair)
    groups = defaultdict(list)
    for row in differences:
        groups[(row["eval_split"], row["comparison"], row["support_size"])].append(row)
    curves = []
    for key, rows in sorted(groups.items()):
        curve = dict(zip(("eval_split", "comparison", "support_size"), key))
        curve["paired_cases"] = len(rows)
        for metric in METRICS:
            estimate = clustered_estimate(rows, metric)
            curve[metric] = estimate["mean"]
            curve[metric + "_ci_low"] = estimate["ci_low"]
            curve[metric + "_ci_high"] = estimate["ci_high"]
            if metric == "success":
                curve.update({name: estimate[name] for name in ("campaigns", "seeds")})
        curves.append(curve)
    return differences, curves, mismatches


def tracked_job_ids(state):
    values = []

    def visit(value):
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in {"id", "job_id", "slurm_job_id"}:
                    visit(child)
                elif isinstance(child, (dict, list, tuple)) or JOB_PATTERN.fullmatch(str(child)):
                    visit(child)
        elif JOB_PATTERN.fullmatch(str(value)):
            values.append(str(value))

    visit(state.get("jobs", []))
    return sorted(set(values))


def gpu_count(alloc_tres):
    fields = {}
    for value in (alloc_tres or "").split(","):
        if "=" in value:
            key, item = value.split("=", 1)
            fields[key] = number(item, 0)
    if "gres/gpu" in fields:
        return int(fields["gres/gpu"])
    return int(sum(value for key, value in fields.items() if key.startswith("gres/gpu:")))


def refresh_accounting(analysis, state, *, enabled=True, now=None):
    """Query only workflow job IDs, at most once every 60 seconds across runs."""
    now = time.time() if now is None else now
    path = Path(analysis) / "slurm_accounting.json"
    old = read_json(path) or {}
    job_ids = tracked_job_ids(state)
    if old and now - number(old.get("query_epoch"), 0) < 60:
        return old
    report = {"query_epoch": now, "queried_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
              "tracked_job_ids": job_ids, "jobs": [], "query_error": None}
    if not enabled or not job_ids:
        report["query_error"] = "accounting disabled" if not enabled else None
        atomic_json(path, report)
        return report
    fields = ("JobID", "JobName", "State", "ExitCode", "ElapsedRaw", "AllocTRES", "Start", "End", "NodeList", "JobIDRaw")
    display_fields = ("JobID%64", "JobName%80", "State%40", *fields[3:])
    try:
        result = subprocess.run(["sacct", "--jobs", ",".join(job_ids), "--noheader", "--parsable2",
                                 "--format", ",".join(display_fields)], text=True, capture_output=True,
                                check=False, timeout=15)
        if result.returncode:
            report["query_error"] = result.stderr.strip() or f"sacct exit {result.returncode}"
        else:
            for line in result.stdout.splitlines():
                values = line.split("|")
                if len(values) < len(fields):
                    continue
                raw = dict(zip(fields, values))
                identifier = raw["JobID"]
                if not ACCOUNTING_PATTERN.fullmatch(identifier):
                    continue  # exclude batch/extern steps to avoid duplicate GPU hours
                if not any(identifier == job or identifier.startswith(job + "_") for job in job_ids):
                    continue
                elapsed = number(raw["ElapsedRaw"], 0)
                gpus = gpu_count(raw["AllocTRES"])
                report["jobs"].append({
                    "job_id": identifier, "name": raw["JobName"], "state": raw["State"],
                    "job_id_raw": raw["JobIDRaw"],
                    "exit_code": raw["ExitCode"], "elapsed_seconds": elapsed,
                    "allocated_gpus": gpus, "allocated_gpu_hours": elapsed * gpus / 3600,
                    "start": raw["Start"], "end": raw["End"], "nodes": raw["NodeList"],
                    "array_range": "[" in identifier,
                })
    except (OSError, subprocess.TimeoutExpired) as error:
        report["query_error"] = f"{type(error).__name__}: {error}"
    identifiers = {row["job_id"] for row in report["jobs"]}
    for row in report["jobs"]:
        row["array_parent"] = any(identifier.startswith(row["job_id"] + "_") for identifier in identifiers)
        if row["array_parent"]:
            row["allocated_gpu_hours"] = 0.0
    atomic_json(path, report)
    return report


def update_marked_section(path, body):
    """Preserve manual content under the same per-journal lock as the root."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = path.read_text() if path.exists() else ""
        block = BEGIN + "\n" + body.rstrip() + "\n" + END
        if old.count(BEGIN) > 1 or old.count(END) > 1 or ((BEGIN in old) != (END in old)):
            raise ValueError(f"journal has unmatched or repeated report markers: {path}")
        if BEGIN in old:
            first = old.index(BEGIN)
            last = old.index(END, first) + len(END)
            revised = old[:first] + block + old[last:]
        else:
            revised = old.rstrip() + "\n\n" + block + "\n"
        atomic_text(path, revised)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def audit_summary(workspace):
    path = Path(workspace) / "2_data_storage/processed/chembl37_redesign_audit/audit.json"
    audit = read_json(path)
    if not audit:
        return {"status": "pending"}
    overlap = audit.get("overlaps", [])
    molecules = set()
    paired_rows = 0
    try:
        with (path.parent / "paired_observations.csv").open() as stream:
            for row in csv.DictReader(stream):
                paired_rows += 1
                if row.get("molecule"):
                    molecules.add(row["molecule"])
    except OSError:
        pass
    return {
        "status": "availability audit complete",
        "release": audit.get("observed_release"),
        "scope": audit.get("scope"),
        "record_counts": audit.get("record_counts", {}),
        "complete_exact_record_counts": audit.get("complete_exact_record_counts", {}),
        "assay_pair_groups": audit.get("overlap_assay_pairs", len(overlap)),
        "paired_record_rows": paired_rows,
        "distinct_measured_molecules": len(molecules),
        "largest_assay_pair_compound_count": max((row.get("paired_compounds", 0) for row in overlap), default=0),
        "supported_current_representation_record_rows": sum(row.get("supported_current_representation", 0) for row in overlap),
        "observed_heavy_atom_min": min((row["heavy_atoms_min"] for row in overlap if "heavy_atoms_min" in row), default=None),
        "observed_heavy_atom_max": max((row["heavy_atoms_max"] for row in overlap if "heavy_atoms_max" in row), default=None),
        "measured_benchmark_status": "dataset audit; no measured transfer benchmark run",
        "artifact": str(path),
    }


def collect_test_reports(workspace):
    directory = Path(workspace) / "5_test/results"
    names = [
        "redesign_integrated_cpu.xml", "redesign_data_worker.xml",
        "redesign_model_cpu.xml", "redesign_sampling_evaluation_cpu.xml",
        "redesign_reporter_cpu.xml", "redesign_search_worker.xml",
        "redesign_adaptation_cpu.xml",
    ]
    paths = [directory / name for name in names]
    paths.extend(sorted(directory.glob("redesign_operation_gpu_*.xml")))
    records = []
    for path in paths:
        try:
            suites = list(ET.parse(path).getroot().iter("testsuite"))
            totals = {key: sum(int(suite.get(key, 0)) for suite in suites)
                      for key in ("tests", "failures", "errors", "skipped")}
        except (OSError, ET.ParseError, ValueError):
            continue
        records.append({"report": path.name,
                        "passed": totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"],
                        **totals})
    return records


def _percent(value):
    return "pending" if number(value) is None else f"{100 * float(value):.1f}%"


def _fmt(value, digits=2):
    return "pending" if number(value) is None else f"{float(value):.{digits}f}"


def _relative_link(path, journal_dir):
    return os.path.relpath(path, journal_dir)


def render_result_section(snapshot, study_root, journal_dir):
    curves = snapshot["curves"]
    controlled = [row for row in curves if row["stage"] == "controlled"]
    lines = [
        "## Automatic experiment record",
        "",
        f"Updated {snapshot['updated_utc']}. Phase: {snapshot['workflow_phase']}.",
        "",
    ]
    if controlled:
        tests = [row for row in controlled if row["eval_split"] == "test"]
        lines.append(f"Completed exact-oracle evaluation batches: {snapshot['controlled_batches']}. "
                     f"Held-out test batches: {snapshot['controlled_test_batches']}. "
                     "Each success meets both requested endpoint requirements.")
        lines.extend(["", "| Model and comparison | Split | N=16 | N=32 | N=64 | N=128 |",
                      "|---|---|---:|---:|---:|---:|"])
        groups = defaultdict(dict)
        for row in controlled:
            groups[(row["architecture"], row["mode"], row["eval_split"])][row["support_size"]] = row
        for (architecture, mode, split), values in sorted(groups.items()):
            label = f"{architecture}: {MODE_NAMES.get(mode, mode)}"
            fields = [_percent(values.get(size, {}).get("success")) for size in SUPPORT_SIZES]
            lines.append(f"| {label} | {split} | " + " | ".join(fields) + " |")
        lines.extend([
            "",
            "Each percentage is the fraction of 16 allocated testing slots that met the complete profile. "
            "A missing selection or failed external test contributes zero. The lead is included in N.",
            "",
            "The CSV curves average leads within each campaign and seed, then campaigns and source seeds. "
            "The 95% bootstrap intervals resample campaigns and source seeds; individual proposals are not "
            "treated as independent campaigns.",
        ])
        paired = [row for row in snapshot["paired_curves"]
                  if row["comparison"] == "conditional:native minus factorized:native"]
        if paired:
            lines.extend(["", "The native architecture comparison uses the same factorized response scorer "
                          "within each paired seed, campaign, lead, and support size.",
                          "", "| Split | Support N | Conditional − factorized success | 95% interval | Matched cases |",
                          "|---|---:|---:|---:|---:|"])
            for row in paired:
                estimate = number(row.get("success"))
                lower, upper = number(row.get("success_ci_low")), number(row.get("success_ci_high"))
                interval = "pending" if lower is None or upper is None else f"{100*lower:.1f} to {100*upper:.1f} points"
                lines.append(f"| {row['eval_split']} | {row['support_size']} | "
                             f"{100*estimate:+.1f} points | {interval} | {row['paired_cases']} |")
        elif tests:
            lines.extend(["", "The paired architecture difference is pending matching native results with "
                          "the same scorer checkpoint."])
        lines.extend([
            "",
            "Pre-ranking yield counts distinct unmeasured successes per attempted proposal; the local "
            "version also requires the declared lead neighbourhood. Efficiency is successes divided by N+16. "
            "The simulator's full-pool oracle calls are extra evaluation costs.",
            "",
            f"Full-pool oracle calls in controlled batches: {snapshot['controlled_full_pool_oracle_calls']:,}.",
        ])
    else:
        lines.append("Controlled comparison results are pending. Operation and pilot records are reported separately.")
    preliminary = [row for row in curves if row["stage"] != "controlled"]
    if preliminary:
        lines.extend(["", "| Preliminary run | Split | Support N | Attempts/testing slots | Joint yield before ranking | Local joint yield | Selected success |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for row in preliminary[:16]:
            matching = [case for case in snapshot["preliminary_cases"]
                        if case["stage"] == row["stage"] and case["architecture"] == row["architecture"]
                        and case["mode"] == row["mode"] and case["support_size"] == row["support_size"]
                        and case["eval_split"] == row["eval_split"]]
            budgets = sorted({f"{case['attempts']}/{case['k']}" for case in matching})
            lines.append(f"| {row['stage']}: {row['architecture']}/{row['mode']} | {row['eval_split']} | "
                         f"{row['support_size']} | {', '.join(budgets)} | "
                         f"{_percent(row['pre_ranking_unmeasured_joint_profile_yield_per_attempt'])} | "
                         f"{_percent(row['pre_ranking_neighbourhood_joint_profile_yield_per_attempt'])} | "
                         f"{_percent(row['success'])} |")
        lines.extend(["", "These rows exercise the integrated pipeline or guide development. Their declared "
                      "attempt and testing budgets are retained in the case CSV."])
    audit = snapshot["chembl_audit"]
    if audit.get("status") != "pending":
        lines.extend([
            "",
            f"The bounded {audit.get('release')} availability audit examined "
            f"{audit.get('record_counts', {}).get('on_target', 0):,} on-target and "
            f"{audit.get('record_counts', {}).get('off_target', 0):,} off-target records. "
            f"It found {audit['assay_pair_groups']} overlapping assay pairs, "
            f"{audit['paired_record_rows']} joined records over {audit['distinct_measured_molecules']} "
            f"distinct molecules, and at most {audit['largest_assay_pair_compound_count']} compounds "
            "in one assay pair.",
            "",
            f"Current-representation coverage: {audit['supported_current_representation_record_rows']} joined rows. "
            f"Observed heavy-atom counts span {audit['observed_heavy_atom_min']} to "
            f"{audit['observed_heavy_atom_max']}. This is a data-availability audit; measured transfer evaluation "
            "will follow dataset and representation preparation.",
        ])
    analysis = Path(study_root) / "analysis"
    lines.extend([
        "",
        f"[Case measurements]({_relative_link(analysis/'case_metrics.csv', journal_dir)}), "
        f"[support-size curves]({_relative_link(analysis/'support_curves.csv', journal_dir)}), "
        f"[paired comparisons]({_relative_link(analysis/'paired_curves.csv', journal_dir)}), "
        f"[current machine-readable report]({_relative_link(analysis/'report.json', journal_dir)}).",
        "",
        "All design-success values above use the exact response oracle. Predicted probabilities are used for ranking.",
    ])
    return "\n".join(lines)


def render_implementation_section(snapshot, study_root, journal_dir):
    completed = [row for row in snapshot["sources"] if row["phase"] == "source_complete"]
    lines = [
        "## Automatic implementation and runtime record", "",
        f"Updated {snapshot['updated_utc']}. Phase: {snapshot['workflow_phase']}.", "",
        f"Source runs with a final summary: {len(completed)} of {len(snapshot['sources'])} discovered runs. "
        f"Committed generation batches: {snapshot['committed_generation_batches']}. "
        f"Externally evaluated batches: {snapshot['evaluated_batches']}.",
    ]
    if snapshot["test_reports"]:
        lines.extend(["", "| Validation report | Passed | Failed/error | Skipped |",
                      "|---|---:|---:|---:|"])
        for row in snapshot["test_reports"]:
            report_path = Path(study_root).parents[2] / "5_test/results" / row["report"]
            lines.append(f"| [{row['report']}]({_relative_link(report_path, journal_dir)}) | "
                         f"{row['passed']} | {row['failures']+row['errors']} | {row['skipped']} |")
        lines.extend(["", "Focused reports overlap with the integrated report; their counts are displayed separately."])
    if snapshot["sources"]:
        lines.extend(["", "| Run | Status | Updates | Total parameters | Source training seconds | Peak Torch GiB |",
                      "|---|---|---:|---:|---:|---:|"])
        for row in snapshot["sources"]:
            parameters = "pending" if row["parameter_count"] is None else f"{int(row['parameter_count']):,}"
            lines.append(f"| {row['run']} | {row['phase']} | {row.get('step') or 0} | {parameters} | "
                         f"{_fmt(row.get('training_seconds'),1)} | {_fmt(row.get('peak_torch_memory_gib'))} |")
    accounting = snapshot["accounting"]
    lines.extend(["", f"Tracked allocated GPU time: {snapshot['allocated_gpu_hours']:.3f} GPU-hours. "
                  "This uses elapsed allocation time multiplied by the allocated GPU count; CPU-only preparation contributes zero."])
    if accounting.get("jobs"):
        lines.extend(["", "| Slurm job | State | Exit | GPUs | Elapsed seconds | GPU-hours |",
                      "|---|---|---|---:|---:|---:|"])
        for row in accounting["jobs"]:
            lines.append(f"| {row['job_id']} | {row['state']} | {row['exit_code']} | "
                         f"{row['allocated_gpus']} | {row['elapsed_seconds']:.0f} | "
                         f"{row['allocated_gpu_hours']:.3f} |")
    known = {row["job_id"] for row in accounting.get("jobs", [])}
    unresolved = [job for job in snapshot["tracked_jobs"] if job not in known
                  and not any(value.startswith(job + "_") for value in known)]
    if unresolved:
        lines.extend(["", "Awaiting an accounting record: " + ", ".join(unresolved) + "."])
    if accounting.get("query_error"):
        lines.extend(["", "Accounting refresh: " + accounting["query_error"] + "."])
    lines.extend([
        "",
        "Accounting is refreshed at intervals of at least 60 seconds and queries only job IDs registered "
        "in workflow_state.json. Failed and pending jobs stay in the report.",
        "",
        "Proposal timing includes transport, the terminal clean prediction, final decoding, host chemistry, "
        "neighbourhood checks, and response scoring. Count/noise preparation and support-state inference "
        "have separate timing records. Invalid, unsupported, duplicate, and excluded proposals remain in the ledgers.",
        "",
        f"[Source and runtime table]({_relative_link(Path(study_root)/'analysis/source_and_compute.csv', journal_dir)}), "
        f"[training history]({_relative_link(Path(study_root)/'analysis/training_history.csv', journal_dir)}), "
        f"[Slurm accounting]({_relative_link(Path(study_root)/'analysis/slurm_accounting.json', journal_dir)}).",
    ])
    return "\n".join(lines)


def make_plots(analysis, curves, paired, history):
    """Scientific figures contain only observed completed summary values."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    analysis = Path(analysis)
    paths = []
    controlled = [row for row in curves if row["stage"] == "controlled"]
    panels = (
        ("success", "Joint-profile success / 16 testing slots"),
        ("pre_ranking_neighbourhood_joint_profile_yield_per_attempt", "Local successes / proposal attempt"),
        ("efficiency", "Successes / allocated compound budget"),
    )
    for split in sorted({row["eval_split"] for row in controlled}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
        groups = defaultdict(list)
        for row in controlled:
            if row["eval_split"] == split:
                groups[(row["architecture"], row["mode"])].append(row)
        for (architecture, mode), rows in sorted(groups.items()):
            rows.sort(key=lambda row: row["support_size"])
            label = f"{architecture}: {MODE_NAMES.get(mode, mode)}"
            for axis, (metric, title) in zip(axes, panels):
                values = [row for row in rows if number(row.get(metric)) is not None]
                if not values:
                    continue
                x = [row["support_size"] for row in values]
                y = [row[metric] for row in values]
                line, = axis.plot(x, y, "o-", label=label, linewidth=2 if mode == "native" else 1)
                if all(number(row.get(metric + "_ci_low")) is not None for row in values):
                    axis.fill_between(x, [row[metric + "_ci_low"] for row in values],
                                      [row[metric + "_ci_high"] for row in values],
                                      alpha=.1, color=line.get_color())
                axis.set_title(title, fontsize=10)
                axis.set_xscale("log", base=2)
                axis.set_xticks(SUPPORT_SIZES, [str(n) for n in SUPPORT_SIZES])
                axis.set_xlabel("Measured support compounds, including lead")
                axis.grid(alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
        fig.suptitle(f"Exact-oracle {split} campaigns — campaign/source-seed bootstrap intervals")
        destination = analysis / f"support_curves_{split}.png"
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        paths.append(destination.name)
    architecture_pairs = [row for row in paired if row["comparison"] == "conditional:native minus factorized:native"]
    if architecture_pairs:
        fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for split in sorted({row["eval_split"] for row in architecture_pairs}):
            values = sorted([row for row in architecture_pairs if row["eval_split"] == split],
                            key=lambda row: row["support_size"])
            x = [row["support_size"] for row in values]
            y = [100 * row["success"] for row in values]
            line, = axis.plot(x, y, "o-", label=split)
            intervals = [row for row in values if number(row.get("success_ci_low")) is not None]
            if intervals:
                axis.errorbar([row["support_size"] for row in intervals],
                    [100 * row["success"] for row in intervals],
                    yerr=np.maximum(0, np.asarray([
                        [100*(row["success"]-row["success_ci_low"]) for row in intervals],
                        [100*(row["success_ci_high"]-row["success"]) for row in intervals]])),
                    fmt="none", capsize=4, color=line.get_color())
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(SUPPORT_SIZES, [str(n) for n in SUPPORT_SIZES])
        axis.set_xlabel("Measured support compounds, including lead")
        axis.set_ylabel("Conditional minus factorized success (percentage points)")
        axis.set_title("Paired native generators with a common factorized scorer")
        axis.grid(alpha=.25)
        axis.legend()
        destination = analysis / "paired_architecture_difference.png"
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        paths.append(destination.name)
    if history:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        groups = defaultdict(list)
        for row in history:
            groups[(row["stage"], row["architecture"], row["seed"], row["run"])].append(row)
        for (stage, architecture, source_seed, _), values in sorted(groups.items(), key=lambda item: str(item[0])):
            values.sort(key=lambda row: row.get("step", 0))
            for axis, metric in zip(axes, ("flow_loss", "response_loss", "decode_loss")):
                observed = [row for row in values if number(row.get(metric)) is not None]
                if observed:
                    axis.plot([row["step"] for row in observed], [row[metric] for row in observed],
                              label=f"{stage}/{architecture}/{source_seed}", linewidth=1)
                axis.set_xlabel("Source optimizer updates")
                axis.set_title(metric.replace("_", " ").capitalize())
                axis.grid(alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
        destination = analysis / "source_training_losses.png"
        fig.savefig(destination, dpi=160)
        plt.close(fig)
        paths.append(destination.name)
    return paths


def run_once(study_root, workspace, journal_dir, *, slurm=True, plots=True):
    study_root, workspace, journal_dir = map(Path, (study_root, workspace, journal_dir))
    analysis = study_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / ".reporter.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = read_json(study_root / "workflow_state.json") or {}
        cases, generations, sources, history, ignored = collect_artifacts(study_root)
        curves = aggregate_curves(cases)
        pairs, paired_curves, pairing_mismatches = paired_differences(cases)
        accounting = refresh_accounting(analysis, state, enabled=slurm)
        job_costs = {}
        for row in accounting.get("jobs", []):
            job_costs[row["job_id"]] = row
            if row.get("job_id_raw"):
                job_costs[row["job_id_raw"]] = row
        for row in sources:
            job = job_costs.get(str(row.get("slurm_job_id")), {})
            row["allocation_state"] = job.get("state")
            row["allocated_gpu_hours"] = job.get("allocated_gpu_hours")
        controlled = [row for row in cases if row["stage"] == "controlled"]
        snapshot = {
            "schema": "emf.redesign.report.v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "workflow_phase": state.get("phase", "waiting for workflow state"),
            "tracked_jobs": tracked_job_ids(state),
            "accounting": accounting,
            "allocated_gpu_hours": sum(row["allocated_gpu_hours"] for row in accounting.get("jobs", [])),
            "sources": sources, "curves": curves, "paired_curves": paired_curves,
            "committed_generation_batches": len(generations), "evaluated_batches": len(cases),
            "controlled_batches": len(controlled),
            "preliminary_cases": [row for row in cases if row["stage"] != "controlled"],
            "controlled_test_batches": sum(row["eval_split"] == "test" for row in controlled),
            "controlled_full_pool_oracle_calls": int(sum(number(row.get("full_pool_outcome_reader_calls"), 0) for row in controlled)),
            "excluded_outcome_sources": ignored, "pairing_mismatches": pairing_mismatches,
            "uncertainty_units": "leads averaged within seed/campaign; bootstrap independent campaigns and source seeds",
            "chembl_audit": audit_summary(workspace),
            "test_reports": collect_test_reports(workspace),
            "agent_status": dict(state.get("agents", {})),
        }
        focused = {row["report"]: row for row in snapshot["test_reports"]}
        for role, name in (("data", "redesign_data_worker.xml"),
                           ("model", "redesign_model_cpu.xml"),
                           ("sampling", "redesign_sampling_evaluation_cpu.xml")):
            if name in focused:
                row = focused[name]
                snapshot["agent_status"][role] = (
                    f"{row['passed']} focused checks passed; "
                    f"{row['failures']+row['errors']} failures/errors"
                )
        cpu = focused.get("redesign_integrated_cpu.xml", {})
        gpu = [row for row in snapshot["test_reports"] if row["report"].startswith("redesign_operation_gpu_")]
        snapshot["integrated_cpu_passed"] = cpu.get("passed")
        snapshot["gpu_operation_passed"] = sum(row["passed"] for row in gpu)
        snapshot["agent_status"]["reviewer"] = (
            f"Integrated CPU checks: {snapshot['integrated_cpu_passed']}; "
            f"allocated-GPU operation checks: {snapshot['gpu_operation_passed']}"
        )
        write_csv(analysis / "case_metrics.csv", cases)
        write_csv(analysis / "generation_metrics.csv", generations)
        write_csv(analysis / "source_and_compute.csv", sources)
        write_csv(analysis / "training_history.csv", history)
        write_csv(analysis / "support_curves.csv", curves)
        write_csv(analysis / "paired_case_differences.csv", pairs)
        write_csv(analysis / "paired_curves.csv", paired_curves)
        snapshot["figures"] = make_plots(analysis, curves, paired_curves, history) if plots else []
        atomic_json(analysis / "report.json", snapshot)
        update_marked_section(journal_dir / "experiment_results_1011.md",
                              render_result_section(snapshot, study_root, journal_dir))
        update_marked_section(journal_dir / "implementation_log_1011.md",
                              render_implementation_section(snapshot, study_root, journal_dir))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return snapshot


def main():
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument("--output-root", type=Path, default=Path("4_experimental_result/result/calibrated_redesign_001"))
    parser.add_argument("--journal-dir", type=Path, default=Path("1_research_file/result_log"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--no-slurm", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    workspace = args.workspace_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else workspace / args.output_root
    journal_dir = args.journal_dir if args.journal_dir.is_absolute() else workspace / args.journal_dir
    while True:
        try:
            report = run_once(output_root, workspace, journal_dir,
                              slurm=not args.no_slurm, plots=not args.no_plots)
            print(json.dumps({"updated_utc": report["updated_utc"],
                              "phase": report["workflow_phase"],
                              "source_runs": len(report["sources"]),
                              "evaluated_batches": report["evaluated_batches"],
                              "controlled_batches": report["controlled_batches"],
                              "integrated_cpu_passed": report["integrated_cpu_passed"],
                              "gpu_operation_passed": report["gpu_operation_passed"],
                              "allocated_gpu_hours": report["allocated_gpu_hours"]}), flush=True)
        except Exception as error:
            print(json.dumps({"report_error": f"{type(error).__name__}: {error}",
                              "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
