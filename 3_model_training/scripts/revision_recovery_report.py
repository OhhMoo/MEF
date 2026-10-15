"""Read-only live and final report for the revision recovery study."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

BOOTSTRAP_SEED = 20261014
BOOTSTRAP_SAMPLES = 2000
STAGES = ("source", "validation", "test", "adapter_validation", "adapter_test", "adapter_train")
ALIASES = {"source": ("source", "source_recovery"),
           "validation": ("validation", "recovery_validation"),
           "test": ("test", "recovery_test"),
           "adapter_validation": ("adapter_validation",),
           "adapter_test": ("adapter_test",), "adapter_train": ("adapter_train",)}
DEPLOYABLE = ("conditional_continue", "conditional_restart", "rollback_25", "rollback_50",
              "adapter_continue", "conditional_sampling_baseline", "prior_sampling_baseline",
              "direct_mutation")


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _num(row, key, default=None):
    value = row.get(key, default)
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _stage_from_task(task, label=""):
    name = str(task.get("stage", label)).lower()
    for stage, aliases in ALIASES.items():
        if name in aliases or label in aliases:
            return stage
    return None


def _task_candidates(study, stage):
    """Return output paths, preferring immutable resolved manifests."""
    tasks_dir = study / "tasks"
    resolved = []
    if tasks_dir.is_dir():
        for path in sorted(tasks_dir.glob("*.resolved.json")):
            label = path.name.removesuffix(".resolved.json")
            if not any(label == alias or label.startswith(alias + "_") for alias in ALIASES[stage]):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, list):
                for task in value:
                    if isinstance(task, dict) and _stage_from_task(task, label) == stage and task.get("output"):
                        resolved.append((Path(task["output"]), task, True))
    if resolved:
        return resolved
    roots = [study / alias for alias in ALIASES[stage]]
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for marker in root.rglob("task_complete.json"):
            relative = marker.relative_to(study)
            if any(part == "operation" or part == "analysis" or part.startswith("source_snapshot") for part in relative.parts):
                continue
            try:
                done = _read_json(marker)
            except ValueError:
                continue
            task = done.get("task", {})
            if _stage_from_task(task, root.name) == stage:
                candidates.append((marker.parent, task, False))
    return candidates


def _complete_runs(study):
    # One completed task per (stage, seed); retries replace earlier outputs.
    picked = {}
    for stage in STAGES:
        for output, task, from_resolved in _task_candidates(study, stage):
            if not output.is_absolute():
                output = (Path.cwd() / output).resolve()
            marker_path = output / "task_complete.json"
            summary_path = output / ("training_summary.json" if stage == "adapter_train" else "summary.json")
            if not marker_path.is_file() or not summary_path.is_file():
                continue
            try:
                marker, summary = _read_json(marker_path), _read_json(summary_path)
            except ValueError:
                continue
            if marker.get("phase") != "complete" or summary.get("phase") != "complete":
                continue
            seed = int(task.get("seed", summary.get("seed", -1)))
            if seed < 0:
                continue
            key = stage, seed
            previous = picked.get(key)
            complete_time = str(marker.get("completed_utc", summary.get("completed_utc", "")))
            if (previous is None or (from_resolved and not previous["resolved"]) or
                    (from_resolved == previous["resolved"] and complete_time > previous["completed_utc"])):
                picked[key] = {"stage": stage, "seed": seed, "path": output,
                               "summary": summary, "resolved": from_resolved,
                               "completed_utc": complete_time}
    return sorted(picked.values(), key=lambda r: (STAGES.index(r["stage"]), r["seed"]))


def _load_run(run):
    path = run["path"]
    result = dict(run)
    for name in ("cases", "roots", "attempts", "selected", "budget_frontier", "feasibility", "reference_witnesses"):
        file = path / f"{name}.csv"
        result[name] = _read_csv(file) if file.is_file() else []
    return result


def _stage_status(study, runs, gates):
    manifest = {}
    path = study / "manifest.json"
    if path.is_file():
        try:
            manifest = _read_json(path).get("task_groups", {})
        except ValueError:
            pass
    status = {}
    for stage in STAGES:
        planned = next((manifest[a] for a in ALIASES[stage] if manifest.get(a)), None)
        expected_count = len(planned) if planned is not None else 0
        expected_seeds = {int(t["seed"]) for t in planned if isinstance(t, dict) and "seed" in t} if planned else set()
        if not expected_seeds:
            tasks_dir = study / "tasks"
            if tasks_dir.is_dir():
                for alias in ALIASES[stage]:
                    file = tasks_dir / f"{alias}.resolved.json"
                    if file.is_file():
                        try:
                            values = json.loads(file.read_text(encoding="utf-8"))
                            expected_seeds.update(int(t["seed"]) for t in values if isinstance(t, dict) and "seed" in t)
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            pass
        completed_seeds = {r["seed"] for r in runs if r["stage"] == stage}
        expected = bool(planned) or bool(expected_seeds) or any((study / a).exists() for a in ALIASES[stage]) or any(
            (study / "tasks" / f"{a}.resolved.json").is_file() for a in ALIASES[stage])
        completed_count = sum(1 for r in runs if r["stage"] == stage)
        complete = bool(expected_seeds) and expected_seeds <= completed_seeds and completed_count >= expected_count
        status[stage] = "complete" if complete else ("pending" if expected else "not scheduled")
    source_gate = gates.get("source_gate")
    if source_gate is not None and source_gate.get("passed") is False:
        for stage in ("adapter_train", "adapter_validation", "adapter_test"):
            status[stage] = "skipped"
    adapter_gate = gates.get("adapter_gate")
    if adapter_gate is not None and adapter_gate.get("passed") is False:
        status["adapter_test"] = "skipped"
    return status


def _expected_seeds(study, stage, observed):
    """Declared task seeds define paired coverage; observations cannot shrink it."""
    path = study / "manifest.json"
    if path.is_file():
        try:
            groups = _read_json(path).get("task_groups", {})
            for alias in ALIASES[stage]:
                tasks = groups.get(alias)
                if tasks:
                    return sorted({int(t["seed"]) for t in tasks if isinstance(t, dict) and "seed" in t})
        except (ValueError, TypeError):
            pass
    return sorted(set(observed))


def _feasibility_map(runs):
    audited = {}
    for run in runs:
        for row in run["feasibility"]:
            both = _truth(row.get("negative_goal_feasible")) and _truth(row.get("positive_goal_feasible"))
            # Current audit stores the distinct counts rather than these convenience booleans.
            if row.get("negative_goal_feasible") in (None, ""):
                both = (_num(row, "negative_goal_feasible_distinct", 0) or 0) > 0 and (_num(row, "positive_goal_feasible_distinct", 0) or 0) > 0
            audited[row.get("campaign_id", "")] = {"feasible": _truth(row.get("feasible_stratum")), "both": both}
    return audited


def _aggregate_cases(runs, audited):
    grouped = {}
    for run in runs:
        split = run["summary"].get("split", run["stage"])
        for row in run["cases"]:
            key = (run["stage"], split, row.get("method", ""), row.get("campaign_id", ""), run["seed"])
            grouped[key] = row
    output = []
    for key, row in grouped.items():
        stage, split, method, campaign, seed = key
        slots = _num(row, "topk") or 0
        record = {"stage": stage, "split": split, "method": method, "campaign_id": campaign,
                  "seed": seed, "successes": _num(row, "topk_successes") or 0,
                  "slots": slots, "total_seconds": _num(row, "total_charged_seconds"),
                  "post_update_seconds": _num(row, "post_update_charged_seconds"),
                  "raw_attempts": _num(row, "raw_candidate_attempts", _num(row, "candidate_attempts", 0)) or 0,
                  "raw_valid": _num(row, "raw_candidate_valid_connected_attempts", _num(row, "raw_valid_connected_attempts", 0)) or 0,
                  "raw_protected": _num(row, "raw_candidate_protected_attempts", _num(row, "raw_protected_copy_attempts", 0)) or 0,
                  "selected_root_validity": _num(row, "selected_root_validity_fraction"),
                  "feasible": audited.get(campaign, {}).get("feasible", False),
                  "both_goals": audited.get(campaign, {}).get("both", False)}
        output.append({**record, "stratum": "all"})
        if record["both_goals"]:
            output.append({**record, "stratum": "verified_both_goals"})
    return output


def _bootstrap_contrasts(study, runs, audited):
    # Unit is campaign; a sampled campaign carries all paired model seeds with it.
    frontiers = []
    for run in runs:
        if run["stage"] not in ("test", "adapter_test"):
            continue
        for row in run["budget_frontier"]:
            frontiers.append({**row, "stage": run["stage"], "seed": run["seed"], "campaign_id": row.get("campaign_id", ""),
                              "budget_basis": row.get("budget_basis", ""),
                              "budget": _num(row, "wall_budget_seconds"),
                              "actual_seconds": _num(row, "charged_seconds_used"),
                              "rate": ((_num(row, "topk_successes", 0) or 0) / (_num(row, "topk") or 1)),
                              "feasible": audited.get(row.get("campaign_id", ""), {}).get("feasible", False),
                              "both_goals": audited.get(row.get("campaign_id", ""), {}).get("both", False)})
    output = []
    methods = sorted({r.get("method") for r in frontiers} & (set(DEPLOYABLE) | {"randomized_suffix"}))
    stages = sorted({r["stage"] for r in frontiers})
    scopes = (("all", lambda r: True), ("verified_both_goals", lambda r: r["both_goals"]))
    for stage in stages:
      seeds = _expected_seeds(study, stage, [r["seed"] for r in frontiers if r["stage"] == stage])
      stage_frontiers = [r for r in frontiers if r["stage"] == stage]
      stage_methods = sorted({r["method"] for r in stage_frontiers} & (set(DEPLOYABLE) | {"randomized_suffix"}))
      for method in stage_methods:
        if method in ("conditional_continue", "conditional_restart"):
            continue
        for baseline in ("conditional_continue", "conditional_restart"):
          for basis in ("total", "post_update"):
            for budget in sorted({r["budget"] for r in stage_frontiers if r["budget_basis"] == basis and r["budget"] is not None}):
              for scope, predicate in scopes:
                        selected = [r for r in stage_frontiers if r["budget_basis"] == basis and r["budget"] == budget and predicate(r)]
                        table = {(r["campaign_id"], r["seed"], r["method"]): r["rate"] for r in selected}
                        campaigns = sorted({r["campaign_id"] for r in selected})
                        diffs = {}
                        coverage = {}
                        for campaign in campaigns:
                            paired_seeds = [seed for seed in seeds if (campaign, seed, method) in table and (campaign, seed, baseline) in table]
                            values = [table[(campaign, seed, method)] - table[(campaign, seed, baseline)]
                                      for seed in paired_seeds]
                            if values:
                                diffs[campaign] = statistics.mean(values)
                                coverage[campaign] = len(paired_seeds) == len(seeds)
                        item = {"stage": stage, "method": method, "baseline": baseline, "budget_basis": basis,
                                "budget_seconds": budget, "stratum": scope,
                                "seeds": seeds, "paired_campaigns": len(diffs),
                                "seed_coverage_complete": bool(diffs) and all(coverage.values()),
                                "mean_method_minus_baseline": statistics.mean(diffs.values()) if diffs else None,
                                "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                                "status": "complete" if len(seeds) >= 3 and len(diffs) >= 1 and all(coverage.values()) else "coverage_incomplete"}
                        if item["status"] == "complete":
                            rng = np.random.default_rng(BOOTSTRAP_SEED)
                            values = np.array(list(diffs.values()), dtype=float)
                            boot = np.array([rng.choice(values, size=len(values), replace=True).mean()
                                             for _ in range(BOOTSTRAP_SAMPLES)])
                            item.update(ci95_low=float(np.quantile(boot, .025)), ci95_high=float(np.quantile(boot, .975)))
                        output.append(item)
    return output, frontiers


def _root_diagnostics(runs):
    roots, selected_by_root, attempts_by_root = [], {}, {}
    for run in runs:
        if run["stage"] != "test":
            continue
        for row in run["selected"]:
            selected_by_root[(run["seed"], row.get("campaign_id"), row.get("root_sample"), row.get("method"))] = row
        for row in run["attempts"]:
            key = (run["seed"], row.get("campaign_id"), row.get("root_sample"))
            attempts_by_root.setdefault(key, []).append(row)
        for row in run["roots"]:
            if not _truth(row.get("diagnostic_tested")) or _truth(row.get("direct_success")):
                continue
            key = (run["seed"], row.get("campaign_id"), row.get("root_sample"))
            attempts = attempts_by_root.get(key, [])
            # Root rows report any success for the 8-trial randomized/adaptive arms.
            random_hit = _truth(row.get("randomized_suffix_any_success")) or any(
                a.get("method") == "randomized_suffix" and _truth(a.get("goal_success")) for a in attempts)
            adaptive_hit = _truth(row.get("oracle_adaptive_any_success")) or any(
                a.get("method") == "oracle_adaptive_suffix" and _truth(a.get("goal_success")) for a in attempts)
            oracle = selected_by_root.get((*key, "oracle_independent_restart"))
            ordinary_selected = selected_by_root.get((*key, "randomized_suffix"))
            oracle_same_pool = selected_by_root.get((*key, "oracle_select_same_suffix_pool"))
            roots.append({"seed": run["seed"], "campaign_id": key[1], "root_sample": key[2],
                          "randomized_8_trial_recovered": random_hit,
                          "adaptive_oracle_8_trial_recovered": adaptive_hit,
                          "ordinary_random_winner_success": bool(ordinary_selected and _truth(ordinary_selected.get("goal_success"))),
                          "oracle_same_pool_winner_success": bool(oracle_same_pool and _truth(oracle_same_pool.get("goal_success"))),
                          "independent_oracle_restart_selected_success": bool(oracle and _truth(oracle.get("goal_success"))),
                          "bounded": (_num(row, "bounded_radius_rms", 99) if _num(row, "bounded_radius_rms", 99) is not None else 99) <= .5})
    return roots


def _pool_diagnostics(runs):
    rows = {}
    tracked = {"randomized_suffix", "oracle_select_same_suffix_pool", "oracle_adaptive_suffix",
               "oracle_independent_restart", "direct_mutation", "oracle_direct_mutation"}
    for run in runs:
        if run["stage"] != "test":
            continue
        for row in run["attempts"]:
            method = row.get("method", "")
            if method not in tracked:
                continue
            acc = rows.setdefault(method, {"attempts": 0, "raw_goal_hits": 0,
                                           "selected_roots": 0, "selected_goal_hits": 0})
            acc["attempts"] += 1
            acc["raw_goal_hits"] += int(_truth(row.get("goal_success")))
        for row in run["selected"]:
            method = row.get("method", "")
            if method in rows:
                rows[method]["selected_roots"] += 1
                rows[method]["selected_goal_hits"] += int(_truth(row.get("goal_success")))
    return rows


def _figures(study, frontiers, roots):
    if not any(r.get("stage") == "test" for r in _complete_runs(study)):
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    analysis = study / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    panel_specs = (("total", "Total charged time (history retained)",
                    {"conditional_continue", "conditional_restart", "rollback_25", "rollback_50", "adapter_continue"}),
                   ("post_update", "Post-update time (fresh-start reference costs shown)", set(DEPLOYABLE) | {"adapter_continue"}))
    for ax, (basis, title, allowed) in zip(axes, panel_specs):
        rows = [r for r in frontiers if r["budget_basis"] == basis and r["method"] in allowed
                and r["budget"] is not None and r["actual_seconds"] is not None
                and (r["stage"] == "test" or (r["stage"] == "adapter_test" and r["method"] == "adapter_continue"))]
        if rows:
            for method in sorted({r["method"] for r in rows}):
                grouped = {}
                for row in rows:
                    if row["method"] == method:
                        grouped.setdefault(row["budget"], []).append(row)
                xs = sorted(grouped)
                actual = [statistics.mean([r["actual_seconds"] for r in grouped[x]]) for x in xs]
                rates = [statistics.mean([r["rate"] for r in grouped[x]]) for x in xs]
                fresh = method in {"direct_mutation", "conditional_sampling_baseline", "prior_sampling_baseline"}
                label = method.replace("_", " ")
                if fresh and basis == "post_update":
                    label += " (fresh-start reference)"
                ax.plot(actual, rates, marker="o", linestyle="--" if fresh else "-", label=label)
            ax.set(xlabel="Actual charged seconds / campaign", ylabel="Top-16 goal successes / slots",
                   ylim=(0, 1), title=title)
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No completed budget-frontier rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    fig.suptitle("Test success versus measured time budget")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        target = analysis / f"success_vs_actual_time.{suffix}"; fig.savefig(target, dpi=180); files.append(str(target.relative_to(study)))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if roots:
        labels = ("Ordinary randomized\nwinner", "Oracle same-pool\nwinner", "Randomized suffix\n8-trial availability",
                  "Adaptive oracle\n8-trial availability", "Independent restart\noracle winner")
        fields = ("ordinary_random_winner_success", "oracle_same_pool_winner_success", "randomized_8_trial_recovered",
                  "adaptive_oracle_8_trial_recovered", "independent_oracle_restart_selected_success")
        rates = [sum(r[f] for r in roots if r["bounded"]) / max(1, sum(r["bounded"] for r in roots)) for f in fields]
        ax.bar(labels, rates, color=("#4c78a8", "#f58518", "#54a24b")); ax.set_ylim(0, 1)
        ax.set_ylabel("Share of tested direct failures recovered"); ax.set_title("Failed-root recovery and goal reversal")
        ax.grid(axis="y", alpha=.25)
    else:
        ax.text(.5, .5, "No completed diagnostic roots", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Failed-root recovery and goal reversal")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        target = analysis / f"failed_root_recoverability.{suffix}"; fig.savefig(target, dpi=180); files.append(str(target.relative_to(study)))
    plt.close(fig)
    return files


def build_report(study):
    study = Path(study).resolve()
    study.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(r) for r in _complete_runs(study)]
    gates = {}
    for name in ("source_gate", "adapter_gate"):
        path = study / f"{name}.json"
        if path.is_file():
            try: gates[name] = _read_json(path)
            except ValueError: pass
    stages = _stage_status(study, runs, gates)
    audited = _feasibility_map(runs)
    cases = _aggregate_cases(runs, audited)
    contrasts, frontiers = _bootstrap_contrasts(study, runs, audited)
    roots = _root_diagnostics(runs)
    pools = _pool_diagnostics(runs)
    figures = _figures(study, frontiers, roots)
    coordinator_complete = False
    status_path = study / "status.json"
    if status_path.is_file():
        try: coordinator_complete = _read_json(status_path).get("phase") in {"complete", "complete_with_recorded_retries"}
        except ValueError: pass
    adapter_done = all(stages[s] in {"complete", "skipped", "not scheduled"}
                       for s in ("adapter_train", "adapter_validation", "adapter_test"))
    if gates.get("source_gate", {}).get("passed") is True:
        adapter_done = (stages["adapter_train"] == "complete" and
                        stages["adapter_validation"] == "complete" and
                        "adapter_gate" in gates and
                        (stages["adapter_test"] in {"complete", "skipped"}))
    report_status = "complete" if coordinator_complete or (stages["test"] == "complete" and adapter_done) else "pending"
    data = {"status": report_status,
            "stages": stages, "completed_runs": [{"stage": r["stage"], "seed": r["seed"],
                "path": str(r["path"].relative_to(study)) if r["path"].is_relative_to(study) else str(r["path"])} for r in runs],
            "case_summaries": cases, "paired_campaign_bootstrap": contrasts,
            "failed_root_diagnostics": roots, "candidate_pool_diagnostics": pools,
            "gates": gates, "figures": figures,
            "feasibility_audited_campaigns": len(audited), "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED}
    (study / "analysis").mkdir(parents=True, exist_ok=True)
    (study / "analysis" / "report.json").write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Revision recovery study", "", f"Study status: **{report_status}**. The report reads completed task artifacts and refreshes as stages finish.", "",
             "## Stage status", "", *[f"- **{stage}:** {stages[stage]}" for stage in STAGES], ""]
    if gates:
        lines += ["## Gates", ""]
        for name, gate in gates.items():
            if name == "source_gate":
                detail = f"{gate.get('recovered_roots', 0)} bounded failed roots across {len(gate.get('campaigns', []))} training campaigns."
            else:
                gain = gate.get("success_gain")
                validity = gate.get("validity_change")
                detail = (f"validation success gain {gain:.3f}; validity change {validity:.3f}."
                          if isinstance(gain, (int, float)) and isinstance(validity, (int, float))
                          else "validation metrics pending.")
            lines.append(f"- **{name.replace('_', ' ')}:** {'passed' if gate.get('passed') else 'not passed'}; {detail}")
        lines.append("")
    final_cases = [r for r in cases if r["stage"] in ("test", "adapter_test")]
    if final_cases:
        lines += ["## Final test results", "", "Success slots show top-16 goal hits divided by available top-16 slots. Raw validity counts include every candidate attempt; selected-root validity describes the one retained candidate per root.", "",
                  "| Stage | Method | Stratum | Goal hits / slots | Raw valid / attempts | Selected-root validity | Charged seconds |",
                  "|---|---|---|---:|---:|---:|---:|"]
        methods = {}
        for row in final_cases:
            key = (row["stage"], row["method"], row["stratum"])
            a = methods.setdefault(key, {k: 0.0 for k in ("successes", "slots", "raw_valid", "raw_attempts", "selected_num", "selected_n", "seconds")})
            for k in ("successes", "slots", "raw_valid", "raw_attempts"):
                a[k] += row[k]
            if row["selected_root_validity"] is not None:
                a["selected_num"] += row["selected_root_validity"]; a["selected_n"] += 1
            if row["total_seconds"] is not None: a["seconds"] += row["total_seconds"]
        for (stage, method, stratum), a in sorted(methods.items()):
            validity = f"{a['raw_valid']:.0f}/{a['raw_attempts']:.0f}" if a["raw_attempts"] else "pending"
            selected = f"{a['selected_num']/a['selected_n']:.3f}" if a["selected_n"] else "pending"
            lines.append(f"| {stage} | {method} | {stratum} | {a['successes']:.0f}/{a['slots']:.0f} | {validity} | {selected} | {a['seconds']:.1f} |")
        lines.append("")
        test_campaigns = {r["campaign_id"] for r in cases if r["stage"] == "test"}
        adapter_campaigns = {r["campaign_id"] for r in cases if r["stage"] == "adapter_test"}
        if adapter_campaigns:
            shared = len(test_campaigns & adapter_campaigns)
            lines.append(f"The adapter test reused {shared} campaign IDs from the prior test set. The adapter acceptance gate used exact validation results; source recoverability was evaluated on training campaigns only.")
            lines.append("")
    if contrasts:
        lines += ["## Paired budget comparisons", "", "Each bootstrap draw resamples campaigns and keeps all model seeds from each sampled campaign together. Effects are method minus the named baseline in top-16 success fraction.", "",
                  "| Stage | Method − baseline | Stratum | Budget basis | Seconds | Paired campaigns | Seeds | Effect (95% interval) |",
                  "|---|---|---|---|---:|---:|---|---:|"]
        for row in contrasts:
            value = "coverage incomplete" if row["status"] != "complete" else f"{row['mean_method_minus_baseline']:.3f} ({row['ci95_low']:.3f}, {row['ci95_high']:.3f})"
            lines.append(f"| {row['stage']} | {row['method']} − {row['baseline']} | {row['stratum']} | {row['budget_basis']} | {row['budget_seconds']:g} | {row['paired_campaigns']} | {', '.join(map(str,row['seeds']))} | {value} |")
        lines.append("")
    if roots:
        denom = sum(r["bounded"] for r in roots)
        lines += ["## Failed direct roots", "", f"Among {denom} bounded, diagnostic-tested roots where direct continuation failed, ordinary randomized winner selection succeeded on {sum(r['ordinary_random_winner_success'] for r in roots)}/{denom}; oracle selection from that same pool succeeded on {sum(r['oracle_same_pool_winner_success'] for r in roots)}/{denom}; randomized suffix found a hit in its eight-trial pool on {sum(r['randomized_8_trial_recovered'] for r in roots)}/{denom}; adaptive oracle found one on {sum(r['adaptive_oracle_8_trial_recovered'] for r in roots)}/{denom}; independent oracle restart selected a goal hit on {sum(r['independent_oracle_restart_selected_success'] for r in roots)}/{denom}. The randomized and adaptive searches use K=8 exact calls per root across the same diagnostic-root set. The denominator includes direct-failed tested roots only; oracle methods use exact outcomes to select or alter proposals.", ""]
        lines.append("The source-gate recovery class is capped at 0.5 editable-coordinate RMS; oracle recovery is a privileged diagnostic.")
        lines.append("")
    if pools:
        lines += ["## Candidate pools and root winners", "",
                  "Raw goal hits count all decoded attempts. Selected goal hits count retained one-winner-per-root rows; direct mutation exposes its unrestricted pool alongside the selected output.", "",
                  "| Method | Raw goal hits / attempts | Selected goal hits / roots |", "|---|---:|---:|"]
        for method, row in sorted(pools.items()):
            lines.append(f"| {method} | {row['raw_goal_hits']}/{row['attempts']} | {row['selected_goal_hits']}/{row['selected_roots']} |")
        lines.append("")
    if figures:
        lines += ["## Figures", "", *[f"- [{Path(f).name}]({f})" for f in figures], ""]
    pending = [stage for stage, value in stages.items() if value == "pending"]
    if pending:
        lines += ["Pending stages: " + ", ".join(pending) + ". Their outcomes are omitted until completed artifacts appear.", ""]
    (study / "live_report.md").write_text("\n".join(lines), encoding="utf-8")
    root = study.parents[2] if (len(study.parents) >= 3 and study.parent.name == "result"
                                and study.parent.parent.name == "4_experimental_result") else None
    if root is not None:
        log = root / "1_research_file" / "result_log" / "experiment_results_1014_3.md"
        log.parent.mkdir(parents=True, exist_ok=True)
        friendly = {"conditional_continue": "Continue from saved state", "conditional_restart": "Restart from original input",
                    "rollback_25": "Roll back 25%", "rollback_50": "Roll back 50%",
                    "adapter_continue": "Learned state adapter", "conditional_sampling_baseline": "Conditional fresh-start sampling",
                    "prior_sampling_baseline": "Prior-model fresh-start sampling", "direct_mutation": "Direct mutation search"}
        local_time = __import__('datetime').datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        log_lines = ["# Revision recovery experiment results (1014_3)", "",
                     f"Updated {local_time}.", "",
                     "A saved state, or root, is the intermediate molecular representation captured when the property goal changes. The goal reversal changes endpoint 1 from an increase target to a decrease target while measurements and the endpoint 2 target stay fixed. Strategies share the original noise and prefix; rollback restores an earlier state, while restart uses the initial noise.", "",
                     "The study asks three questions: does rolling back before continuing help; can independent suffix trials recover roots that direct continuation misses, and can an oracle identify promising proposals; can a learned state adapter improve continuation on validation and test campaigns? Each success is one of 16 retained output slots. Raw validity counts every candidate attempt; selected-root validity describes the retained candidate per root.", "",
                     "The full-cap method table uses each method's configured proposal budget, and those budgets differ. The time curves enforce the same wall-time cap per comparison and plot actual charged time; a point below its cap shows time left when the next complete batch did not fit. Full-cap success counts therefore describe each method's configured run, while the curves compare success under matched time limits.", "",
                     f"Main test stage: **{stages['test']}**. Adapter test: **{stages['adapter_test']}**.", ""]
        if final_cases:
            log_lines += ["## Final test results", "",
                          "| Stage | Strategy | Audited set | Goal hits / 16-slot opportunities | Raw valid / attempts | Selected-root validity | Total charged seconds |",
                          "|---|---|---|---:|---:|---:|---:|"]
            groups = {}
            for row in final_cases:
                key = (row["stage"], row["method"], row["stratum"])
                acc = groups.setdefault(key, {"success": 0, "slots": 0, "valid": 0, "attempts": 0, "sel": [], "seconds": 0})
                acc["success"] += row["successes"]; acc["slots"] += row["slots"]
                acc["valid"] += row["raw_valid"]; acc["attempts"] += row["raw_attempts"]
                if row["selected_root_validity"] is not None: acc["sel"].append(row["selected_root_validity"])
                if row["total_seconds"] is not None: acc["seconds"] += row["total_seconds"]
            for (stage, method, stratum), a in sorted(groups.items()):
                sv = f"{statistics.mean(a['sel']):.3f}" if a["sel"] else "pending"
                log_lines.append(f"| {stage.replace('_', ' ')} | {friendly.get(method, method.replace('_', ' '))} | {stratum.replace('_', ' ')} | {a['success']:.0f}/{a['slots']:.0f} | {a['valid']:.0f}/{a['attempts']:.0f} | {sv} | {a['seconds']:.1f} |")
            test_methods = {key[1]: value for key, value in groups.items() if key[0] == "test" and key[2] == "all"}
            eligible = [(method, values["success"] / values["slots"], values) for method, values in test_methods.items()
                        if method in DEPLOYABLE and values["slots"]]
            direct = test_methods.get("conditional_continue")
            if eligible and direct:
                best_method, _, best = max(eligible, key=lambda item: item[1])
                log_lines += ["", "## What the test comparison shows", "",
                              f"Across the completed test cases, {friendly.get(best_method, best_method)} had {best['success']:.0f}/{best['slots']:.0f} goal hits, compared with {direct['success']:.0f}/{direct['slots']:.0f} for continuing from the saved state. This is the highest observed rate among the deployable test methods in these campaigns and model seeds."]
                paired = [x for x in contrasts if x["stage"] == "test" and x["stratum"] == "all" and x["baseline"] == "conditional_continue" and x["budget_basis"] == "post_update" and x["status"] == "complete"]
                if paired:
                    cap = max(x["budget_seconds"] for x in paired)
                    contrast = max((x for x in paired if x["budget_seconds"] == cap), key=lambda x: x["mean_method_minus_baseline"])
                    log_lines.append(f"At the largest matched post-update budget ({cap:g} seconds), {friendly.get(contrast['method'], contrast['method'])} changed the paired success fraction by {contrast['mean_method_minus_baseline']:+.3f} versus continuation (95% campaign-bootstrap interval {contrast['ci95_low']:+.3f} to {contrast['ci95_high']:+.3f}).")
        else:
            log_lines.append("Final test cases are pending; no test counts are reported yet.")
        if gates:
            log_lines += ["", "## Research gates", ""]
            if "source_gate" in gates:
                gate = gates["source_gate"]
                log_lines.append(f"The source-only training gate {'passed' if gate.get('passed') else 'did not pass'}: it found {gate.get('recovered_roots', 0)} bounded direct-failed roots across {len(gate.get('campaigns', []))} training campaigns. This decides whether the adapter is trained; it does not measure test performance.")
            if "adapter_gate" in gates:
                gate = gates["adapter_gate"]
                log_lines.append(f"The exact validation gate {'passed' if gate.get('passed') else 'did not pass'}: adapter success changed by {gate.get('success_gain', 0):+.3f} per slot and validity changed by {gate.get('validity_change', 0):+.3f}. The test evaluates this fixed validation decision.")
        pending_stages = [s.replace('_', ' ') for s, state in stages.items() if state == "pending"]
        log_lines += ["", "## Remaining work", ""]
        if pending_stages:
            log_lines.append("Pending stages: " + ", ".join(pending_stages) + ".")
        elif data["status"] == "complete":
            log_lines.append("All required stages for the recorded gate decisions are complete.")
        else:
            log_lines.append("The coordinator has not completed all required or gate-selected stages.")
        log_lines += ["", f"Detailed live report: [live_report.md](../../4_experimental_result/result/{study.name}/live_report.md).", ""]
        log.write_text("\n".join(log_lines), encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="4_experimental_result/result/revision_recovery_001")
    args = parser.parse_args()
    report = build_report(args.study)
    print(json.dumps({"status": report["status"], "completed_runs": len(report["completed_runs"]),
                      "report": str(Path(args.study) / "live_report.md")}, indent=2))


if __name__ == "__main__":
    main()
