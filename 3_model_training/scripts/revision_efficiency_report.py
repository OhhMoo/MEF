"""Build a read-only summary from primary revision-efficiency outputs.

The reader consumes only completed run summary.json, cases.csv,
budget_frontier.csv, cost_batches.csv, and the explicit static_b/seed*/
metrics.csv inputs. It never reads source snapshots, attempts ledgers, manifests,
status files, or its own derived analysis directory.
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

import numpy as np


DEFAULT_STUDY = Path("4_experimental_result/result/revision_efficiency_001")
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_SAMPLES = 2000
SEED_SET = (142, 143, 144)

CASE_FIELDS = {
    "campaign_id", "scenario", "method", "revision_target", "seed", "split",
    "attempts", "valid_connected_attempts", "eligible_attempts",
    "distinct_eligible", "distinct_goal_successes", "success_attempts",
    "endpoint2_preserved_attempts", "topk", "topk_filled", "topk_successes",
    "topk_success_fraction", "total_charged_seconds",
    "incremental_charged_seconds", "charged_forward_nfe",
    "charged_backward_nfe", "incremental_forward_nfe",
    "incremental_backward_nfe",
}
FRONTIER_FIELDS = {
    "campaign_id", "scenario", "method", "revision_target", "budget_basis", "wall_budget_seconds",
    "charged_seconds_used", "unused_budget_seconds", "cap_limited", "attempts",
    "valid_connected_attempts", "eligible_attempts", "distinct_eligible",
    "distinct_goal_successes", "success_attempts", "endpoint2_preserved_attempts",
    "topk", "topk_filled", "topk_successes", "topk_success_fraction",
    "total_charged_seconds", "incremental_charged_seconds",
}
BATCH_FIELDS = {
    "campaign_id", "scenario", "method", "revision_target", "batch_first",
    "batch_count", "total_charged_seconds", "incremental_charged_seconds",
    "cumulative_charged_seconds", "cumulative_incremental_seconds",
    "charged_forward_nfe", "cumulative_charged_forward_nfe",
    "charged_backward_nfe", "cumulative_charged_backward_nfe",
}
STATIC_FIELDS = {
    "campaign_id", "method", "support_size", "evidence", "seed",
    "forward_sample_nfe", "backward_sample_nfe", "seconds_including_chemistry",
    "attempts", "valid", "protected", "eligible_distinct", "distinct_successes",
    "success_attempts", "topk", "topk_filled", "topk_successes",
    "topk_success_fraction",
}


class ReportSchemaError(ValueError):
    """A completed primary artifact is missing required fields or values."""


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportSchemaError(f"Cannot read primary JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportSchemaError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(required - fields)
            if missing:
                raise ReportSchemaError(
                    f"{path} is missing required columns: {', '.join(missing)}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise ReportSchemaError(f"Cannot read primary CSV {path}: {exc}") from exc
    except csv.Error as exc:
        raise ReportSchemaError(f"Malformed primary CSV {path}: {exc}") from exc


def _number(row: dict, field: str, *, integer: bool = False) -> int | float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportSchemaError(f"Invalid numeric {field}={row.get(field)!r}") from exc
    if not math.isfinite(value):
        raise ReportSchemaError(f"Non-finite {field}={row.get(field)!r}")
    if integer:
        if not value.is_integer():
            raise ReportSchemaError(f"Expected integer {field}, found {value}")
        return int(value)
    return value


def _bool(row: dict, field: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ReportSchemaError(f"Invalid boolean {field}={row.get(field)!r}")


def _target(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportSchemaError(f"Invalid revision_target={value!r}") from exc
    if not math.isfinite(number):
        raise ReportSchemaError(f"Non-finite revision_target={value!r}")
    return number


def _skip_directory(relative: Path) -> bool:
    return any(part == "analysis" or part.startswith("source_snapshot")
               for part in relative.parts)


def _summary_paths(study: Path):
    """Discover only one-level validation/test run folders, never operation."""
    for split in ("validation", "test"):
        split_root = study / split
        if not split_root.is_dir():
            continue
        for run_dir in sorted(split_root.iterdir()):
            if run_dir.is_dir() and not _skip_directory(Path(split) / run_dir.name):
                summary = run_dir / "summary.json"
                if summary.is_file():
                    yield summary


def _completed_time(summary: dict, path: Path) -> datetime:
    value = summary.get("completed_utc")
    if not isinstance(value, str) or not value.strip():
        raise ReportSchemaError(f"Completed run lacks completed_utc: {path}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReportSchemaError(f"Invalid completed_utc in {path}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_case(row: dict[str, str], summary: dict, source: Path) -> dict:
    item = dict(row)
    for field in ("attempts", "valid_connected_attempts", "eligible_attempts",
                  "distinct_eligible", "distinct_goal_successes", "success_attempts",
                  "endpoint2_preserved_attempts", "topk", "topk_filled",
                  "topk_successes", "charged_forward_nfe", "charged_backward_nfe",
                  "incremental_forward_nfe", "incremental_backward_nfe"):
        item[field] = _number(row, field, integer=True)
    for field in ("topk_success_fraction", "total_charged_seconds",
                  "incremental_charged_seconds"):
        item[field] = _number(row, field)
    item["revision_target"] = _target(row["revision_target"])
    item["seed"] = _number(row, "seed", integer=True)
    if item["seed"] != int(summary["seed"]) or item["split"] != summary["split"]:
        raise ReportSchemaError(f"Case metadata disagrees with run summary: {source}")
    if item["attempts"] < item["valid_connected_attempts"]:
        raise ReportSchemaError(f"Validity count exceeds attempts: {source}")
    if item["topk_filled"] > item["topk"] or item["topk_successes"] > item["topk_filled"]:
        raise ReportSchemaError(f"Selection counts are inconsistent: {source}")
    return item


def _normalize_frontier(row: dict[str, str], summary: dict, source: Path) -> dict:
    item = dict(row)
    for field in ("attempts", "valid_connected_attempts", "eligible_attempts",
                  "distinct_eligible", "distinct_goal_successes", "success_attempts",
                  "endpoint2_preserved_attempts", "topk", "topk_filled",
                  "topk_successes"):
        item[field] = _number(row, field, integer=True)
    for field in ("wall_budget_seconds", "charged_seconds_used", "unused_budget_seconds",
                  "topk_success_fraction", "total_charged_seconds",
                  "incremental_charged_seconds"):
        item[field] = _number(row, field)
    item["cap_limited"] = _bool(row, "cap_limited")
    item["revision_target"] = _target(row["revision_target"])
    if row["budget_basis"] not in {"total", "incremental"}:
        raise ReportSchemaError(f"Invalid budget_basis={row['budget_basis']!r}: {source}")
    item["budget_basis"] = row["budget_basis"]
    item["seed"] = int(summary["seed"])
    item["split"] = summary["split"]
    if item["topk_filled"] > item["topk"] or item["topk_successes"] > item["topk_filled"]:
        raise ReportSchemaError(f"Budget selection counts are inconsistent: {source}")
    return item


def _normalize_batch(row: dict[str, str], summary: dict, source: Path) -> dict:
    item = dict(row)
    for field in ("batch_first", "batch_count", "charged_forward_nfe",
                  "cumulative_charged_forward_nfe", "charged_backward_nfe",
                  "cumulative_charged_backward_nfe"):
        item[field] = _number(row, field, integer=True)
    for field in ("total_charged_seconds", "incremental_charged_seconds",
                  "cumulative_charged_seconds", "cumulative_incremental_seconds"):
        item[field] = _number(row, field)
    item["revision_target"] = _target(row["revision_target"])
    item["seed"] = int(summary["seed"])
    item["split"] = summary["split"]
    return item


def _collect(study: Path) -> tuple[list[dict], list[dict], list[dict], list[str], list[dict], list[dict]]:
    candidates, pending = [], []
    for summary_path in _summary_paths(study):
        summary = _read_json(summary_path)
        if summary.get("stage") != "revision_efficiency":
            continue
        if summary.get("phase") != "complete":
            continue
        for field in ("seed", "split", "phase", "stage"):
            if field not in summary:
                raise ReportSchemaError(f"{summary_path} lacks summary field {field}")
        run_dir = summary_path.parent
        cases_path = run_dir / "cases.csv"
        frontier_path = run_dir / "budget_frontier.csv"
        batches_path = run_dir / "cost_batches.csv"
        expect_frontier = bool(summary.get("wall_budgets_seconds"))
        required_paths = [cases_path, batches_path]
        if expect_frontier:
            required_paths.append(frontier_path)
        absent = [str(path.relative_to(study)) for path in required_paths if not path.exists()]
        if absent:
            pending.append(str(run_dir.relative_to(study)))
            continue
        case_rows = _read_csv(cases_path, CASE_FIELDS)
        frontier_rows = _read_csv(frontier_path, FRONTIER_FIELDS) if frontier_path.exists() else []
        batch_rows = _read_csv(batches_path, BATCH_FIELDS)
        normalized_cases = [_normalize_case(row, summary, cases_path) for row in case_rows]
        normalized_frontier = [_normalize_frontier(row, summary, frontier_path)
                               for row in frontier_rows]
        normalized_batches = [_normalize_batch(row, summary, batches_path)
                              for row in batch_rows]
        candidates.append({"path": run_dir.relative_to(study),
                           "completed_utc": _completed_time(summary, summary_path),
                           "summary": summary, "cases": normalized_cases,
                           "frontier": normalized_frontier, "batches": normalized_batches,
                           "frontier_status": "available" if frontier_path.exists()
                           else "not_scheduled"})

    # Keep distinct campaign shards, and select the latest complete attempt for
    # a campaign when a retry emitted another run folder for the same seed.
    winner = {}
    for candidate in candidates:
        summary = candidate["summary"]
        for campaign in {row["campaign_id"] for row in candidate["cases"]}:
            key = (summary["split"], int(summary["seed"]), campaign)
            rank = (candidate["completed_utc"], str(candidate["path"]))
            if key not in winner or rank > winner[key][0]:
                winner[key] = (rank, candidate["path"])

    runs, cases, frontier, batches, superseded = [], [], [], [], []
    for candidate in candidates:
        path = candidate["path"]
        summary = candidate["summary"]
        split, seed = summary["split"], int(summary["seed"])
        selected = lambda row: winner.get((split, seed, row["campaign_id"]), (None, None))[1] == path
        selected_cases = [row for row in candidate["cases"] if selected(row)]
        selected_frontier = [row for row in candidate["frontier"] if selected(row)]
        selected_batches = [row for row in candidate["batches"] if selected(row)]
        cases.extend(selected_cases)
        frontier.extend(selected_frontier)
        batches.extend(selected_batches)
        selected_campaigns = len({row["campaign_id"] for row in selected_cases})
        candidate_campaigns = {row["campaign_id"] for row in candidate["cases"]}
        replaced = sorted(campaign for campaign in candidate_campaigns
                          if not selected({"campaign_id": campaign}))
        if replaced:
            superseded.append({"path": str(path), "split": split, "seed": seed,
                               "campaigns_replaced": len(replaced),
                               "selected_run_paths": sorted({str(winner[(split, seed, campaign)][1])
                                                              for campaign in replaced})})
        if candidate_campaigns and not selected_campaigns:
            continue
        runs.append({"path": str(path), "seed": seed, "split": split,
                     "variant": summary.get("variant"),
                     "campaigns": selected_campaigns,
                     "reported_campaigns": int(summary.get("campaigns", 0)),
                     "campaigns_replaced_by_retry": len(replaced),
                     "cases": len(selected_cases), "frontier_rows": len(selected_frontier),
                     "frontier_status": candidate["frontier_status"],
                     "wall_budgets_seconds": summary.get("wall_budgets_seconds", []),
                     "physical_exact_response_calls": summary.get(
                         "physical_exact_response_calls"),
                     "seconds_including_chemistry": summary.get("seconds_including_chemistry")})
    return runs, cases, frontier, batches, pending, superseded


def _group_key(row: dict, include_seed: bool = True) -> tuple:
    fields = [row["split"], row["scenario"], row["method"], row["revision_target"]]
    if include_seed:
        fields.append(row["seed"])
    return tuple(fields)


def _case_summaries(cases: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in cases:
        groups.setdefault(_group_key(row), []).append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        attempts = sum(row["attempts"] for row in rows)
        topk = sum(row["topk"] for row in rows)
        output.append(dict(split=key[0], scenario=key[1], method=key[2],
                           revision_target=key[3], seed=key[4], campaigns=len(rows),
                           attempts=attempts,
                           valid_connected_attempts=sum(row["valid_connected_attempts"] for row in rows),
                           validity_fraction=(sum(row["valid_connected_attempts"] for row in rows)
                                              / attempts if attempts else None),
                           distinct_eligible=sum(row["distinct_eligible"] for row in rows),
                           topk_successes=sum(row["topk_successes"] for row in rows),
                           topk_slots=topk,
                           topk_success_fraction=sum(row["topk_successes"] for row in rows)
                           / topk if topk else None,
                           unused_slots=sum(row["topk"]-row["topk_filled"] for row in rows),
                           total_charged_seconds=sum(row["total_charged_seconds"] for row in rows),
                           incremental_charged_seconds=sum(
                               row["incremental_charged_seconds"] for row in rows),
                           total_forward_nfe=sum(row["charged_forward_nfe"] for row in rows),
                           total_backward_nfe=sum(row["charged_backward_nfe"] for row in rows),
                           incremental_forward_nfe=sum(row["incremental_forward_nfe"] for row in rows),
                           incremental_backward_nfe=sum(row["incremental_backward_nfe"] for row in rows)))
    return output


def _budget_summaries(frontier: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in frontier:
        key = (*_group_key(row), row["budget_basis"], row["wall_budget_seconds"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        attempts = sum(row["attempts"] for row in rows)
        topk = sum(row["topk"] for row in rows)
        output.append(dict(split=key[0], scenario=key[1], method=key[2],
                           revision_target=key[3], seed=key[4],
                           budget_basis=key[5], wall_budget_seconds=key[6], campaigns=len(rows),
                           attempts=attempts,
                           valid_connected_attempts=sum(row["valid_connected_attempts"] for row in rows),
                           validity_fraction=(sum(row["valid_connected_attempts"] for row in rows)
                                              / attempts if attempts else None),
                           topk_successes=sum(row["topk_successes"] for row in rows),
                           topk_slots=topk,
                           topk_success_fraction=(sum(row["topk_successes"] for row in rows)
                                                  / topk if topk else None),
                           unused_slots=sum(row["topk"]-row["topk_filled"] for row in rows),
                           total_charged_seconds=sum(row["total_charged_seconds"] for row in rows),
                           incremental_charged_seconds=sum(
                               row["incremental_charged_seconds"] for row in rows),
                           charged_seconds_used=sum(row["charged_seconds_used"] for row in rows),
                           unused_budget_seconds=sum(row["unused_budget_seconds"] for row in rows),
                           cap_limited_cases=sum(row["cap_limited"] for row in rows)))
    return output


def _paired_bootstrap(frontier: list[dict], *, samples: int = BOOTSTRAP_SAMPLES,
                      rng_seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """Paired continue-minus-restart CIs, resampling campaign clusters only."""
    by_key = {}
    for row in frontier:
        if row["method"] not in {"continue_control", "restart_control",
                                 "conditional_continue", "conditional_restart"}:
            continue
        key = (row["split"], row["scenario"], row["revision_target"],
               row["budget_basis"], row["wall_budget_seconds"], row["seed"],
               row["campaign_id"], row["method"])
        if key in by_key:
            raise ReportSchemaError(f"Duplicate frontier row for {key}")
        by_key[key] = row
    pair_methods = (("continue_control", "restart_control"),
                    ("conditional_continue", "conditional_restart"))
    groups = sorted({(key[0], key[1], key[2], key[3]) for key in by_key},
                    key=lambda item: tuple(str(value) for value in item))
    rng = np.random.default_rng(rng_seed)
    output = []
    for split, scenario, target, basis in groups:
        for continuation, restart in pair_methods:
            available_seeds = sorted({key[5] for key in by_key
                                      if key[:4] == (split, scenario, target, basis)
                                      and key[7] in {continuation, restart}})
            if not available_seeds:
                continue
            budget_sets = []
            for seed in available_seeds:
                for method in (continuation, restart):
                    budget_sets.append({key[4] for key in by_key
                                        if key[:4] == (split, scenario, target, basis)
                                        and key[5] == seed and key[7] == method})
            common_budgets = sorted(set.intersection(*budget_sets)) if budget_sets else []
            for budget in common_budgets:
                setting = (split, scenario, target, basis, budget)
                if not any(key[:5] == setting for key in by_key):
                    continue
                cluster_effects = {}
                campaign_seeds = {}
                for key, continue_row in by_key.items():
                    if key[:5] != setting or key[7] != continuation:
                        continue
                    restart_key = (*key[:7], restart)
                    restart_row = by_key.get(restart_key)
                    if restart_row is None:
                        continue
                    campaign = key[6]
                    seed = key[5]
                    effect = (continue_row["topk_successes"] / continue_row["topk"]
                              - restart_row["topk_successes"] / restart_row["topk"])
                    cluster_effects.setdefault(campaign, {})[seed] = effect
                    campaign_seeds.setdefault(campaign, set()).add(seed)
                expected = set(SEED_SET)
                paired_campaigns = sorted(campaign for campaign, seeds in campaign_seeds.items()
                                          if seeds == expected)
                if available_seeds != list(SEED_SET) or len(paired_campaigns) < 2:
                    output.append(dict(split=split, scenario=scenario,
                                       revision_target=target, budget_basis=basis,
                                       wall_budget_seconds=budget,
                                       continuation=continuation, restart=restart,
                                       paired_campaigns=len(paired_campaigns),
                                       seeds=available_seeds,
                                       status="pending_three_seed_campaign_pairs",
                                       bootstrap_samples=0, mean_continue_minus_restart=None,
                                       ci95_low=None, ci95_high=None))
                    continue
                values = np.asarray([[cluster_effects[campaign][seed]
                                      for seed in SEED_SET]
                                     for campaign in paired_campaigns], dtype=np.float64).mean(axis=1)
                indices = rng.integers(0, len(values), size=(samples, len(values)))
                draws = values[indices].mean(axis=1)
                output.append(dict(split=split, scenario=scenario,
                                   revision_target=target, budget_basis=basis,
                                   wall_budget_seconds=budget,
                                   continuation=continuation, restart=restart,
                                   paired_campaigns=len(paired_campaigns), seeds=list(SEED_SET),
                                   status="complete", bootstrap_samples=samples,
                                   mean_continue_minus_restart=float(values.mean()),
                                   ci95_low=float(np.quantile(draws, .025)),
                                   ci95_high=float(np.quantile(draws, .975))))
    return output


def _static_b(study: Path) -> tuple[list[dict], list[str]]:
    static_root = study / "static_b"
    paths = sorted(static_root.glob("seed*/metrics.csv")) if static_root.exists() else []
    if not paths:
        return [], ["static_b/seed*/metrics.csv"]
    rows = []
    for path in paths:
        try:
            folder_seed = int(path.parent.name.removeprefix("seed"))
        except ValueError as exc:
            raise ReportSchemaError(f"Static B directory must be named seed<id>: {path.parent}") from exc
        for source_row in _read_csv(path, STATIC_FIELDS):
            item = dict(source_row)
            for field in ("support_size", "seed", "forward_sample_nfe", "backward_sample_nfe",
                          "attempts", "valid", "protected", "eligible_distinct",
                          "distinct_successes", "success_attempts", "topk", "topk_filled",
                          "topk_successes"):
                item[field] = _number(source_row, field, integer=True)
            for field in ("seconds_including_chemistry", "topk_success_fraction"):
                item[field] = _number(source_row, field)
            if item["attempts"] < item["valid"] or item["topk_filled"] > item["topk"]:
                raise ReportSchemaError(f"Inconsistent static B metrics in {path}")
            if item["seed"] != folder_seed:
                raise ReportSchemaError(f"Static B row seed disagrees with directory in {path}")
            item["path"] = str(path.relative_to(study))
            rows.append(item)
    return rows, []


def _static_b_summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["seed"], row["method"], row["support_size"], row["evidence"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        attempts = sum(row["attempts"] for row in group)
        valid = sum(row["valid"] for row in group)
        topk = sum(row["topk"] for row in group)
        output.append(dict(seed=key[0], method=key[1], support_size=key[2], evidence=key[3],
                           campaigns=len(group), attempts=attempts, valid=valid,
                           validity_fraction=valid / attempts if attempts else None,
                           topk_successes=sum(row["topk_successes"] for row in group),
                           topk_slots=topk,
                           topk_success_fraction=(sum(row["topk_successes"] for row in group)
                                                  / topk if topk else None),
                           seconds_including_chemistry=sum(
                               row["seconds_including_chemistry"] for row in group),
                           forward_sample_nfe=sum(row["forward_sample_nfe"] for row in group),
                           backward_sample_nfe=sum(row["backward_sample_nfe"] for row in group)))
    return output


def _report_json(study: Path, runs, cases, frontier, batches, pending, superseded,
                 static_rows, static_pending) -> dict:
    coverage = {split: sorted({row["seed"] for row in runs if row["split"] == split})
                for split in ("validation", "test")}
    missing_seed_splits = {split: sorted(set(SEED_SET) - set(seeds))
                           for split, seeds in coverage.items()
                           if set(seeds) != set(SEED_SET)}
    status = ("pending" if not runs else
              "complete" if not pending and not static_pending and not missing_seed_splits
              else "partial")
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "study": str(study), "status": status,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_unit": "campaign_id; all seeds 142, 143, and 144 remain in each sampled cluster",
            "completed_runs": runs, "incomplete_run_directories": pending,
            "seed_coverage": coverage, "missing_seed_splits": missing_seed_splits,
            "missing_static_b_inputs": static_pending,
            "static_b_seed_metrics": static_rows,
            "static_b_summaries_by_seed": _static_b_summaries(static_rows),
            "case_summaries_by_seed": _case_summaries(cases),
            "budget_summaries_by_seed": _budget_summaries(frontier),
            "paired_campaign_bootstrap": _paired_bootstrap(frontier),
            "cost_batch_rows": batches,
            "retry_resolution": {"rule": "For duplicate split/seed/campaign outputs, keep the latest complete summary by completed_utc; retain disjoint campaign shards.",
                                 "superseded_run_directories": superseded},
            "superseded_run_directories": superseded,
            "input_policy": "Only immediate validation/test run summaries and their cases/frontier/cost-batch CSVs, plus explicit static_b run_control metrics, are read. Operation runs, source snapshots, attempts ledgers, manifests, status files, and analysis outputs are excluded."}


def _markdown(report: dict) -> str:
    lines = ["# Revision efficiency study report", "",
             f"Generated: {report['generated_utc']}", "",
             f"Study status: **{report['status']}**.", ""]
    if not report["completed_runs"]:
        lines.append("No complete revision-efficiency run summaries are available yet."
                     " Results will appear after a task writes its primary summary and CSVs.")
    if report["incomplete_run_directories"]:
        lines.extend(["", "Run summaries missing one or more primary CSVs:", ""])
        lines.extend(f"- `{path}`" for path in report["incomplete_run_directories"])
    if report["missing_seed_splits"]:
        lines.extend(["", "Pending seed coverage:"])
        lines.extend(f"- {split}: missing seeds {seeds}"
                     for split, seeds in report["missing_seed_splits"].items())
    if report["missing_static_b_inputs"]:
        lines.extend(["", "Static B run_control input is pending at `static_b/seed*/metrics.csv`."])
    lines.extend(["", "## Case summaries by seed", "",
                  "Success uses the fixed top-k slot denominator; unused slots count as failures. "
                  "Validity is valid connected attempts divided by all attempts.", "",
                  "| Split | Scenario | Method | Update | Seed | Campaigns | Successes / slots | Success | Validity | Unused slots | Total seconds | Incremental seconds |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["case_summaries_by_seed"]:
        lines.append("| {split} | {scenario} | {method} | {target} | {seed} | {campaigns} | {hits}/{slots} | {success:.1%} | {valid:.1%} | {unused} | {total:.2f} | {incremental:.2f} |".format(
            split=row["split"], scenario=row["scenario"], method=row["method"],
            target="all" if row["revision_target"] is None else row["revision_target"],
            seed=row["seed"], campaigns=row["campaigns"], hits=row["topk_successes"],
            slots=row["topk_slots"], success=row["topk_success_fraction"] or 0.,
            valid=row["validity_fraction"] or 0., unused=row["unused_slots"],
            total=row["total_charged_seconds"], incremental=row["incremental_charged_seconds"]))
    lines.extend(["", "## Completed-batch budget frontiers by seed", "",
                  "Only common submitted budgets are listed per run. Cap-limited rows remain visible; no success values are interpolated.", "",
                  "| Split | Scenario | Method | Update | Seed | Basis | Budget (s) | Campaigns | Successes / slots | Validity | Unused slots | Cap-limited cases | Total seconds | Incremental seconds |",
                  "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["budget_summaries_by_seed"]:
        lines.append("| {split} | {scenario} | {method} | {target} | {seed} | {basis} | {budget:g} | {campaigns} | {hits}/{slots} | {valid:.1%} | {unused} | {caps} | {total:.2f} | {incremental:.2f} |".format(
            split=row["split"], scenario=row["scenario"], method=row["method"],
            target="all" if row["revision_target"] is None else row["revision_target"],
            seed=row["seed"], basis=row["budget_basis"], budget=row["wall_budget_seconds"],
            campaigns=row["campaigns"], hits=row["topk_successes"], slots=row["topk_slots"],
            valid=row["validity_fraction"] or 0., unused=row["unused_slots"],
            caps=row["cap_limited_cases"], total=row["total_charged_seconds"],
            incremental=row["incremental_charged_seconds"]))
    lines.extend(["", "## Paired continue–restart bootstrap", "",
                  "Each interval resamples campaign IDs; all three seeds stay together within each campaign. "
                  "Intervals are shown only when the same campaign has paired results for seeds 142–144.", "",
                  "| Split | Scenario | Update | Basis | Wall budget (s) | Continuation | Restart | Paired campaigns | Difference | 95% CI | Status |",
                  "|---|---|---:|---|---:|---|---|---:|---:|---:|---|"])
    for row in report["paired_campaign_bootstrap"]:
        estimate = row["mean_continue_minus_restart"]
        ci = ("—" if row["ci95_low"] is None else
              f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]")
        lines.append("| {split} | {scenario} | {target} | {basis} | {budget:g} | {cont} | {restart} | {n} | {estimate} | {ci} | {status} |".format(
            split=row["split"], scenario=row["scenario"],
            target="all" if row["revision_target"] is None else row["revision_target"],
            basis=row["budget_basis"],
            budget=row["wall_budget_seconds"], cont=row["continuation"],
            restart=row["restart"], n=row["paired_campaigns"],
            estimate="—" if estimate is None else f"{estimate:.3f}", ci=ci,
            status=row["status"]))
    lines.extend(["", "## Reading the cost and coverage fields", "",
                  "The JSON contains per-seed budget summaries with cap-limited campaign counts, unused selection slots, unused wall budget, validity, and total/incremental charged seconds. "
                  "A budget row uses only completed indivisible batches. The report does not interpolate beyond a run cap.", "",
                  "## Static B context", ""])
    if report["static_b_seed_metrics"]:
        lines.extend(["Static B summaries by seed, method, support size, and evidence:", "",
                      "| Seed | Method | Support | Evidence | Campaigns | Successes / slots | Success | Validity | Seconds |",
                      "|---:|---|---:|---|---:|---:|---:|---:|---:|"])
        for row in report["static_b_summaries_by_seed"]:
            lines.append("| {seed} | {method} | {support} | {evidence} | {campaigns} | {hits}/{slots} | {success:.1%} | {valid:.1%} | {seconds:.2f} |".format(
                seed=row["seed"], method=row["method"], support=row["support_size"],
                evidence=row["evidence"], campaigns=row["campaigns"],
                hits=row["topk_successes"], slots=row["topk_slots"],
                success=row["topk_success_fraction"] or 0.,
                valid=row["validity_fraction"] or 0.,
                seconds=row["seconds_including_chemistry"]))
    else:
        lines.append("Static B `run_control` metrics are pending.")
    lines.append("")
    return "\n".join(lines)


def build_report(study_root: Path | str = DEFAULT_STUDY) -> dict:
    study = Path(study_root)
    runs, cases, frontier, batches, pending, superseded = _collect(study)
    static_rows, static_pending = _static_b(study)
    report = _report_json(study, runs, cases, frontier, batches, pending,
                          superseded, static_rows, static_pending)
    analysis = study / "analysis"
    _atomic_json(analysis / "report.json", report)
    _atomic_text(study / "live_report.md", _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY,
                        help="revision_efficiency_001 result directory")
    args = parser.parse_args()
    report = build_report(args.study)
    print(json.dumps({"status": report["status"], "completed_runs": len(report["completed_runs"]),
                      "case_groups": len(report["case_summaries_by_seed"]),
                      "paired_comparisons": len(report["paired_campaign_bootstrap"]),
                      "report": str(args.study / "live_report.md")}))


if __name__ == "__main__":
    main()
