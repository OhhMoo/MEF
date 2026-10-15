"""Run the paired overnight EMF study from explicit, reproducible tasks.

Each seed supplies one frozen representation to six branches. Branches receive
the same training-update time budget, then the same initial sampling noise at
five evaluation budgets. Slurm submission belongs to the coordinator; this
module only prepares, executes, and reports individual tasks.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from emf.evaluation import generate_from_checkpoint
from emf.train import PROJECT_ROOT, atomic_json, load_config, project_path, run_training

CONTROL_DIR = Path(__file__).resolve().parent
STUDY_DIR = PROJECT_ROOT / "4_experimental_result/result/overnight_20261010"
CONFIG_DIR = PROJECT_ROOT / "3_model_training/configs/overnight_20261010"
SEEDS = (42, 43, 44)
LAMBDAS = (0.0, 0.1, 1.0)
NFE_GRID = (8, 16, 32, 64, 128)
GENERATION_SEED = 314159


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scheduler_context():
    return {key: os.environ[key] for key in (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")
        if key in os.environ}


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _representation_context(path, seed):
    path = Path(path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["phase"] != "flow" or payload["step"] != 0:
        raise ValueError("A shared representation must be the flow step-zero checkpoint")
    if payload["config"]["seed"] != seed:
        raise ValueError("Shared representation seed differs from the task seed")
    return {"path": str(path), "sha256": sha256(path), "seed": seed,
            "phase": payload["phase"], "step": payload["step"],
            "data_hash": payload["data_hash"], "model": payload["config"]["model"]}


def prepare(*, study_dir=STUDY_DIR, config_dir=CONFIG_DIR,
            baseline_config=PROJECT_ROOT / "3_model_training/configs/explore.yaml",
            baseline_representation=PROJECT_ROOT / "4_experimental_result/result/exploration_001/checkpoints/representation.pt",
            flow_budget_seconds=3600, max_flow_steps=250000,
            generation_attempts=10000):
    """Write the exact 3 × 3 × 2 design and three uniform warm-up configs."""
    if not math.isfinite(flow_budget_seconds) or flow_budget_seconds <= 0 or max_flow_steps < 1 or generation_attempts < 1:
        raise ValueError("Training budget, update cap, and generation count must be positive")
    study_dir, config_dir = project_path(study_dir), project_path(config_dir)
    base = asdict(load_config(baseline_config))
    initial = _representation_context(baseline_representation, 42)
    if initial["model"] != base["model"]:
        raise ValueError("Baseline representation architecture differs from the configuration")
    data_dir = project_path(base["data_dir"])
    sources = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
               for directory in (PROJECT_ROOT / "3_model_training/emf", CONTROL_DIR)
               for path in sorted(directory.glob("*.py"))}
    shared = {str(seed): str(study_dir / "representations" / f"seed_{seed}" /
                             "checkpoints/representation.pt") for seed in SEEDS}
    study = {
        "format_version": 1, "seeds": list(SEEDS), "lambda_mid": list(LAMBDAS),
        "decoder_modes": ["trainable", "frozen_common"],
        "flow_time_budget_seconds": flow_budget_seconds,
        "max_flow_steps": max_flow_steps,
        "budget_scope": "accumulated synchronized training-update wall time; excludes validation, checkpoint writing, and generation",
        "generation_attempts": generation_attempts, "generation_seed": GENERATION_SEED,
        "generation_batch_size": base["batch_size"], "nfe_grid": list(NFE_GRID),
        "reference_data_dir": str(data_dir),
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "train_npz_sha256": sha256(data_dir / "train.npz"),
        "validation_npz_sha256": sha256(data_dir / "validation.npz"),
        "shared_representations": shared, "initial_representation": initial,
        "config_dir": str(config_dir), "study_dir": str(study_dir),
        "initial_checkpoint_grid": str(study_dir / "initial_checkpoint_grid"),
        "source_sha256": sources,
        "source_snapshot": str(study_dir / "source_snapshot"),
    }
    study_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    study_file = study_dir / "study.json"
    if study_file.exists() and json.loads(study_file.read_text()) != study:
        raise FileExistsError("Study design already exists with different settings; choose a new study directory")
    if study_file.exists() and (study_dir / "tasks.json").exists():
        tasks = json.loads((study_dir / "tasks.json").read_text())
        if any(sha256(task["config"]) != task["config_sha256"] for task in tasks):
            raise ValueError("Prepared task configuration changed; do not rewrite an active study")
        return study, tasks
    atomic_json(study_file, study)
    for relative in sources:
        destination = Path(study["source_snapshot"]) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)

    tasks = []
    for seed in SEEDS:
        warmup = {**base, "seed": seed, "flow_steps": 0, "sample_every": 0,
                  "num_samples": 0, "freeze_decoder": False,
                  "flow_time_budget_seconds": None,
                  "output_dir": str(study_dir / "representations" / f"seed_{seed}")}
        Path(warmup["output_dir"]).mkdir(parents=True, exist_ok=True)
        (config_dir / f"representation_{seed}.yaml").write_text(
            yaml.safe_dump(warmup, sort_keys=False))
        for coefficient in LAMBDAS:
            for decoder_mode in study["decoder_modes"]:
                index = len(tasks)
                tag = f"seed_{seed}_lambda_{coefficient:g}_{decoder_mode}"
                output = study_dir / "runs" / tag
                output.mkdir(parents=True, exist_ok=True)
                config = {**base, "seed": seed, "output_dir": str(output),
                          "lambda_mid": coefficient,
                          "freeze_decoder": decoder_mode == "frozen_common",
                          "flow_steps": max_flow_steps,
                          "flow_time_budget_seconds": flow_budget_seconds,
                          "checkpoint_every": 10000, "sample_every": 10000,
                          "log_every": 100, "num_samples": 1000}
                config_path = config_dir / f"task_{index:02d}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                tasks.append({"index": index, "seed": seed,
                              "lambda_mid": coefficient, "decoder_mode": decoder_mode,
                              "config": str(config_path), "config_sha256": sha256(config_path),
                              "representation": shared[str(seed)], "output_dir": str(output)})
    atomic_json(study_dir / "tasks.json", tasks)
    _write_csv(study_dir / "tasks.csv", tasks)
    return study, tasks


def _load_study(study_dir):
    root = project_path(study_dir)
    return root, json.loads((root / "study.json").read_text())


def _check_dataset(study):
    root = Path(study["reference_data_dir"])
    for filename, key in (("manifest.json", "dataset_manifest_sha256"),
                          ("train.npz", "train_npz_sha256"),
                          ("validation.npz", "validation_npz_sha256")):
        if sha256(root / filename) != study[key]:
            raise ValueError(f"Study dataset changed after preparation: {filename}")


def _check_sources(study):
    for relative, expected in study["source_sha256"].items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"Study source changed after preparation: {relative}")


def prepare_representation(seed, *, study_dir=STUDY_DIR):
    """Train one representation with the pinned source, independently of its six arms."""
    root, study = _load_study(study_dir)
    if seed not in SEEDS:
        raise ValueError("Shared representation seed must be 42, 43, or 44")
    _check_dataset(study)
    _check_sources(study)
    config = load_config(Path(study["config_dir"]) / f"representation_{seed}.yaml")
    output = Path(config.output_dir)
    status = output / "representation_provenance.json"
    if status.exists():
        raise FileExistsError("Representation already attempted; inspect its provenance before explicit recovery")
    context = {"seed": seed, "started_at": utc_now(), "scheduler": scheduler_context(),
               "config": asdict(config), "state": "running"}
    atomic_json(status, context)
    try:
        model, result = run_training(config)
        del model
        context.update({"state": "complete", "result": result,
                        "checkpoint": _representation_context(study["shared_representations"][str(seed)], seed)})
    except Exception as error:
        context.update({"state": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        context["finished_at"] = utc_now()
        atomic_json(status, context)
    return context


def sample_grid(checkpoint, output_dir, *, reference_data_dir,
                num_samples=10000, nfe_grid=NFE_GRID, seed=GENERATION_SEED,
                batch_size=128, device="cuda", context=None):
    """Use identical count/noise draws in every NFE arm; preserve every attempt."""
    from study_metrics import analyze_samples

    checkpoint = Path(checkpoint).resolve(strict=True)
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "grid_manifest.json"
    if manifest_file.exists():
        raise FileExistsError("Sampling grid already attempted; choose a fresh output directory")
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "num_samples": num_samples, "nfe_grid": list(nfe_grid), "seed": seed,
                "batch_size": batch_size, "device": device, "started_at": utc_now(),
                "scheduler": scheduler_context(), "context": context or {}, "state": "running"}
    atomic_json(manifest_file, manifest)
    try:
        summaries = []
        for nfe in nfe_grid:
            destination = output / f"nfe_{nfe:03d}"
            _, summary = generate_from_checkpoint(
                checkpoint, destination, num_samples=num_samples, nfe=nfe,
                seed=seed, device=device, batch_size=batch_size)
            if summary["metadata"]["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during the sampling grid")
            quality = analyze_samples(destination, reference_data_dir)
            summaries.append({"nfe": nfe, **{k: v for k, v in summary.items()
                                             if not isinstance(v, dict)}, **quality})
        _write_csv(output / "grid_summary.csv", summaries)
        manifest["state"] = "complete"
    except Exception as error:
        manifest.update({"state": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        manifest["finished_at"] = utc_now()
        atomic_json(manifest_file, manifest)
    return summaries


def run_task(index, *, study_dir=STUDY_DIR):
    root, study = _load_study(study_dir)
    tasks = json.loads((root / "tasks.json").read_text())
    if index < 0 or index >= len(tasks):
        raise ValueError(f"Task index must be between 0 and {len(tasks) - 1}")
    task = tasks[index]
    _check_dataset(study)
    _check_sources(study)
    if sha256(task["config"]) != task["config_sha256"]:
        raise ValueError("Task configuration changed after study preparation")
    representation = _representation_context(task["representation"], task["seed"])
    initial = study["initial_representation"]
    if representation["data_hash"] != initial["data_hash"] or representation["model"] != initial["model"]:
        raise ValueError("Shared representations must have the same architecture and training identities")
    provenance = json.loads((Path(task["representation"]).parents[1] /
                             "representation_provenance.json").read_text())
    if provenance["state"] != "complete" or provenance["checkpoint"] != representation:
        raise ValueError("Shared representation differs from its completed preparation record")
    output = Path(task["output_dir"])
    status_file = output / "task_provenance.json"
    if status_file.exists():
        raise FileExistsError("Task already attempted; inspect its provenance before explicit recovery")
    context = {"task": task, "representation": representation, "state": "training",
               "started_at": utc_now(), "scheduler": scheduler_context(),
               "study_manifest_sha256": sha256(root / "study.json"),
               "workflow_source_sha256": sha256(__file__)}
    atomic_json(status_file, context)
    try:
        model, result = run_training(load_config(task["config"]), resume=task["representation"])
        context["training_result"] = result
        if result.get("stop_reason") != "time_budget":
            raise RuntimeError("Arm reached its update ceiling before the matched training-time budget")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_step = result["flow_steps"]
        checkpoint = output / "checkpoints" / f"flow_{final_step:07d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError("Training did not write the final immutable sampling checkpoint")
        context.update({"state": "sampling", "training_result": result,
                        "training_finished_at": utc_now(), "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint)})
        atomic_json(status_file, context)
        _check_dataset(study)
        _check_sources(study)
        sample_grid(checkpoint, output / "final_grid",
                    reference_data_dir=study["reference_data_dir"],
                    num_samples=study["generation_attempts"], nfe_grid=study["nfe_grid"],
                    seed=study["generation_seed"], batch_size=study["generation_batch_size"],
                    device=load_config(task["config"]).device, context=task)
        context["state"] = "complete"
    except Exception as error:
        context.update({"state": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        context["finished_at"] = utc_now()
        atomic_json(status_file, context)
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Write the paired task table and configs")
    prep.add_argument("--study-dir", default=str(STUDY_DIR))
    prep.add_argument("--config-dir", default=str(CONFIG_DIR))
    prep.add_argument("--flow-budget-seconds", type=float, default=3600)
    prep.add_argument("--max-flow-steps", type=int, default=250000)
    prep.add_argument("--generation-attempts", type=int, default=10000)
    rep = sub.add_parser("prepare-representation", help="Train one shared representation with the pinned source")
    rep.add_argument("--seed", type=int, required=True, choices=SEEDS)
    rep.add_argument("--study-dir", default=str(STUDY_DIR))
    task = sub.add_parser("run-task", help="Execute one Slurm array task")
    task.add_argument("--index", type=int, required=True)
    task.add_argument("--study-dir", default=str(STUDY_DIR))
    grid = sub.add_parser("sample-grid", help="Evaluate an existing checkpoint at five NFE budgets")
    grid.add_argument("--checkpoint", required=True)
    grid.add_argument("--output-dir", required=True)
    grid.add_argument("--reference-data-dir", default=str(PROJECT_ROOT / "2_data_storage/processed/qm9"))
    grid.add_argument("--num-samples", type=int, default=10000)
    grid.add_argument("--seed", type=int, default=GENERATION_SEED)
    grid.add_argument("--batch-size", type=int, default=128)
    grid.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    report = sub.add_parser("aggregate", help="Report seed-paired effects and allocated GPU hours")
    report.add_argument("--study-dir", default=str(STUDY_DIR))
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        study, tasks = prepare(**args)
        print(json.dumps({"study_dir": study["study_dir"], "tasks": len(tasks)}, indent=2))
    elif command == "prepare-representation":
        print(json.dumps(prepare_representation(**args), indent=2))
    elif command == "run-task":
        print(json.dumps(run_task(**args), indent=2))
    elif command == "sample-grid":
        print(json.dumps(sample_grid(**args), indent=2))
    else:
        from study_analysis import aggregate
        print(json.dumps(aggregate(**args), indent=2))


if __name__ == "__main__":
    main()
