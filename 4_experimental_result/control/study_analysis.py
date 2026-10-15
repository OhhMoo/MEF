"""Report paired scientific effects separately from scheduler resource costs."""

import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from overnight import LAMBDAS, SEEDS, STUDY_DIR, _load_study, _write_csv, atomic_json, utc_now


def _flat(summary):
    return {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}


def _run_record(task, provenance):
    result = provenance.get("training_result", {})
    return {"task_index": task["index"], "seed": task["seed"],
            "lambda_mid": task["lambda_mid"], "decoder_mode": task["decoder_mode"],
            "state": provenance.get("state", "not_started"),
            "flow_steps": result.get("flow_steps"),
            "flow_training_seconds": result.get("flow_training_seconds"),
            "training_runtime_gpu_hours": result.get("runtime_gpu_hours_this_invocation"),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "representation_sha256": provenance.get("representation", {}).get("sha256"),
            "scheduler_job_id": provenance.get("scheduler", {}).get("SLURM_JOB_ID"),
            "output_dir": task["output_dir"], "error": provenance.get("error")}


def paired_effects(records):
    """Target minus control per seed, then descriptive summaries across seeds.

    Three seeds support a sensitivity check, not a precise significance claim.
    Null conditional-distribution distances remain absent from numeric deltas.
    """
    identifiers = {"task_index", "seed", "lambda_mid", "nfe", "flow_steps",
                   "flow_training_seconds", "training_runtime_gpu_hours", "batch_size",
                   "T", "attempts", "generation_seed", "heldout_reference_molecules"}
    groups = defaultdict(dict)
    for row in records:
        groups[(row["seed"], row["nfe"])][(row["lambda_mid"], row["decoder_mode"])] = row
    deltas = []
    for (seed, nfe), arms in sorted(groups.items()):
        comparisons = []
        for mode in ("trainable", "frozen_common"):
            comparisons.extend((f"lambda_{coefficient:g}_vs_0_{mode}",
                                (coefficient, mode), (0.0, mode)) for coefficient in LAMBDAS[1:])
        comparisons.extend((f"trainable_vs_frozen_common_lambda_{coefficient:g}",
                            (coefficient, "trainable"), (coefficient, "frozen_common"))
                           for coefficient in LAMBDAS)
        for label, target_key, control_key in comparisons:
            if target_key not in arms or control_key not in arms:
                continue
            target, control = arms[target_key], arms[control_key]
            if (target.get("attempts") != control.get("attempts")
                    or target.get("generation_seed") != control.get("generation_seed")):
                raise ValueError("A paired comparison requires the same attempts and generation seed")
            for metric in sorted((target.keys() | control.keys()) - identifiers):
                if metric.startswith("heldout_") and not metric.endswith(("wasserstein_1", "js_nats")):
                    continue
                absent = 0 if metric.startswith("failure_") and metric.endswith("_count") else None
                before, after = control.get(metric, absent), target.get(metric, absent)
                if (isinstance(before, (int, float)) and not isinstance(before, bool)
                        and isinstance(after, (int, float)) and not isinstance(after, bool)
                        and math.isfinite(before) and math.isfinite(after)):
                    deltas.append({"comparison": label, "seed": seed, "nfe": nfe,
                                   "metric": metric, "control_value": before,
                                   "target_value": after, "delta": after - before})
    grouped = defaultdict(list)
    for row in deltas:
        grouped[(row["comparison"], row["nfe"], row["metric"])].append(row["delta"])
    summaries = []
    for (comparison, nfe, metric), values in sorted(grouped.items()):
        sd = statistics.stdev(values) if len(values) > 1 else None
        summaries.append({"comparison": comparison, "nfe": nfe, "metric": metric,
                          "seed_pairs": len(values), "mean_delta": statistics.mean(values),
                          "sample_sd": sd, "standard_error": sd / math.sqrt(len(values)) if sd is not None else None})
    return deltas, summaries


def _same_update_records(tasks):
    """Use only update indices available in all six branches of the same seed."""
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["seed"]].append(task)
    records = []
    for seed, arms in grouped.items():
        if len(arms) != 6:
            continue
        samples = []
        timing = []
        for task in arms:
            output = Path(task["output_dir"])
            summaries = {}
            for path in output.glob("samples_*/summary.json"):
                value = json.loads(path.read_text())
                summaries[value["metadata"]["flow_step"]] = value
            samples.append(summaries)
            training_times = {}
            if (output / "training.csv").exists():
                with (output / "training.csv").open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row["phase"] == "flow":
                            training_times[int(row["step"])] = row.get("training_seconds")
            timing.append(training_times)
        common = set.intersection(*(set(values) for values in samples))
        for step in sorted(common):
            for task, summaries, times in zip(arms, samples, timing):
                summary = summaries[step]
                records.append({"task_index": task["index"], "seed": seed,
                                "lambda_mid": task["lambda_mid"], "decoder_mode": task["decoder_mode"],
                                "flow_steps": step, "flow_training_seconds": times.get(step),
                                "generation_seed": summary["seed"], **_flat(summary)})
                records[-1]["seed"] = seed  # Summary seed is the generation seed.
    return records


def scheduler_costs(job_ids):
    """sacct allocation time includes startup, training, sampling, and teardown."""
    if not job_ids:
        return [], {"available": False, "reason": "No scheduler job IDs recorded"}
    command = ["sacct", "--jobs", ",".join(sorted(job_ids)), "--noheader", "--parsable2",
               "--format", "JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,ExitCode"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        return [], {"available": False, "command": command, "error": repr(error)}
    if result.returncode:
        return [], {"available": False, "command": command, "error": result.stderr.strip()}
    costs = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or "." in fields[0]:
            continue  # Job steps share the parent allocation; never double count.
        job_id, name, state, elapsed, allocated, exit_code = fields[:6]
        resources = dict(part.split("=", 1) for part in allocated.split(",") if "=" in part)
        count = resources.get("gres/gpu")
        if count is None:
            typed = [int(value) for key, value in resources.items() if key.startswith("gres/gpu:")]
            count = sum(typed) if typed else 0
        count = int(count)
        seconds = int(elapsed)
        costs.append({"job_id": job_id, "job_name": name, "state": state,
                      "elapsed_seconds": seconds, "allocated_gpu_count": count,
                      "allocated_gpu_hours": count * seconds / 3600,
                      "alloc_tres": allocated, "exit_code": exit_code})
    return costs, {"available": True, "command": command,
                   "allocated_gpu_hours": sum(row["allocated_gpu_hours"] for row in costs)}


def aggregate(*, study_dir=STUDY_DIR):
    root, study = _load_study(study_dir)
    tasks = json.loads((root / "tasks.json").read_text())
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs, final, job_ids = [], [], set()
    for task in tasks:
        path = Path(task["output_dir"]) / "task_provenance.json"
        provenance = json.loads(path.read_text()) if path.exists() else {}
        run = _run_record(task, provenance)
        runs.append(run)
        if run["scheduler_job_id"]:
            job_ids.add(run["scheduler_job_id"])
        if provenance.get("state") != "complete":
            continue
        for nfe in study["nfe_grid"]:
            directory = Path(task["output_dir"]) / "final_grid" / f"nfe_{nfe:03d}"
            summary = json.loads((directory / "summary.json").read_text())
            quality = json.loads((directory / "quality_metrics.json").read_text())
            if (summary["attempts"] != study["generation_attempts"]
                    or summary["seed"] != study["generation_seed"]
                    or summary["metadata"]["checkpoint_sha256"] != provenance["checkpoint_sha256"]):
                raise ValueError("Completed task generation differs from the study sampling/provenance contract")
            final.append({**run, **_flat(summary), **_flat(quality),
                          "seed": task["seed"], "generation_seed": summary["seed"]})
    for seed in SEEDS:
        path = root / "representations" / f"seed_{seed}" / "representation_provenance.json"
        if path.exists():
            job = json.loads(path.read_text()).get("scheduler", {}).get("SLURM_JOB_ID")
            if job:
                job_ids.add(job)
    initial_grid = Path(study["initial_checkpoint_grid"])
    benchmark = []
    if (initial_grid / "grid_manifest.json").exists():
        manifest = json.loads((initial_grid / "grid_manifest.json").read_text())
        if manifest.get("scheduler", {}).get("SLURM_JOB_ID"):
            job_ids.add(manifest["scheduler"]["SLURM_JOB_ID"])
        for path in sorted(initial_grid.glob("nfe_*/summary.json")):
            summary = json.loads(path.read_text())
            quality_path = path.parent / "quality_metrics.json"
            quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
            benchmark.append({**_flat(summary), **_flat(quality)})
    _write_csv(output / "runs.csv", runs)
    _write_csv(output / "quality_and_yield_vs_nfe.csv", final)
    _write_csv(output / "initial_checkpoint_vs_nfe.csv", benchmark)
    deltas, effects = paired_effects(final)
    _write_csv(output / "paired_seed_deltas.csv", deltas)
    _write_csv(output / "paired_effect_summary.csv", effects)
    same_update = _same_update_records(tasks)
    _write_csv(output / "same_update_quality.csv", same_update)
    same_effects = []
    for step in sorted({row["flow_steps"] for row in same_update}):
        _, values = paired_effects([row for row in same_update if row["flow_steps"] == step])
        same_effects.extend({"flow_steps": step, **row} for row in values)
    _write_csv(output / "same_update_paired_effects.csv", same_effects)
    costs, accounting = scheduler_costs(job_ids)
    _write_csv(output / "scheduler_gpu_hours.csv", costs)
    report = {"updated_at": utc_now(), "planned_tasks": len(tasks),
              "complete_tasks": sum(row["state"] == "complete" for row in runs),
              "failed_tasks": sum(row["state"] == "failed" for row in runs),
              "final_nfe_records": len(final), "scheduler": accounting,
              "interpretation": [
                  "Primary comparisons use equal accumulated training-update wall time and identical final count/noise draws.",
                  "Matching update wall time on the same GPU model does not match FLOPs or the number of optimizer updates.",
                  "Same-update comparisons use periodic samples shared across all six branches of a seed; training compute can differ.",
                  "Three paired seeds give descriptive variability, not a precise statistical significance claim.",
                  "Conditional distribution distances compare connected valid attempts with held-out validation molecules.",
                  "Scheduler allocation hours include startup and all generation; training-update budgets do not.",
              ]}
    atomic_json(output / "summary.json", report)
    return report
