"""Execute one pinned task in the bounded evidence-steering study."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .redesign_objectives import redesign_loss
from .redesign_train import (atomic_json, digest, load_redesign_config,
                             project_path, run_training)
from .steering_data import SteeringDataset


def privileged_callbacks(data_dir):
    """Only the explicitly labelled simulator diagnostic can call this."""
    private = json.loads((project_path(data_dir) / "private/oracle.json").read_text())
    coefficients = private["coefficients"]

    def privileged_objective(model, batch, *, episodes, **kwargs):
        states = torch.zeros((len(episodes), model.redesign_config.state_dim),
                             device=batch.baseline.device, dtype=torch.float32)
        states[:, 0] = torch.tensor([coefficients[e.campaign_id] for e in episodes],
                                   device=states.device)
        return redesign_loss(model, batch,
                             flow_state_override=states.index_select(0, batch.episode_index),
                             **kwargs)

    def privileged_state(model, state, campaign_id, dataset):
        result = torch.zeros_like(state)
        result[:, 0] = coefficients[campaign_id]
        return result

    return privileged_objective, privileged_state


def train(config, *, privileged=False):
    objective = privileged_callbacks(config.data_dir)[0] if privileged else None
    return run_training(config, dataset_class=SteeringDataset, objective=objective,
                        goal_rule="fixed outcome-independent improvement .25, preservation -.10",
                        save_step_checkpoints=True)


def selected_checkpoint(manifest, name):
    selection = json.loads((Path(manifest["study_root"]) / "checkpoint_selection.json").read_text())
    return selection["checkpoints"][name]


def evaluate(config, checkpoint, scorer, *, privileged=False, operation=False):
    from .steering_evaluation import run_evaluation
    callback = privileged_callbacks(config.data_dir)[1] if privileged else None
    output = project_path(config.output_dir)
    validation = replace(config, num_attempts=16 if operation else 64,
                         support_sizes=[4] if operation else [4, 16, 64],
                         nfe=8 if operation else config.nfe)
    val = run_evaluation(validation, checkpoint, "validation",
                         output / "evaluation_validation", scorer_checkpoint=scorer,
                         flow_state_override=callback)
    if operation:
        return {"validation": val}
    test = run_evaluation(config, checkpoint, "test", output / "evaluation_test",
                          scorer_checkpoint=scorer, flow_state_override=callback)
    return {"validation": val, "test": test}


def execute(manifest_path, stage, index):
    manifest = json.loads(Path(manifest_path).read_text())
    for path, expected in manifest["snapshot_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"pinned source changed: {path}")
    for path, expected in manifest.get("input_sha256", {}).items():
        if digest(path) != expected:
            raise RuntimeError(f"pinned study input changed: {path}")
    task = manifest["stages"][stage]["tasks"][index]
    print(json.dumps({"event": "start", "stage": stage, "index": index, "task": task,
                      "job_id": os.environ.get("SLURM_JOB_ID")}), flush=True)
    if stage == "trace":
        from .steering_trace import main as trace_main
        sys.argv = [sys.argv[0], *task["arguments"]]
        trace_main()
        return
    config = load_redesign_config(task["config"])
    if stage in {"privileged_train", "privileged_evaluation"}:
        selection = json.loads((Path(manifest["study_root"]) / "checkpoint_selection.json").read_text())
        config = replace(config, train_steps=selection["step"])
    if stage in {"source", "privileged_train", "operation"}:
        report = train(config, privileged=stage == "privileged_train")
        if stage == "operation":
            checkpoint = str(project_path(config.output_dir) / "checkpoints/source_final.pt")
            report["evaluation"] = evaluate(config, checkpoint, checkpoint, operation=True)
            if not report["frozen_representation_unchanged"]:
                raise RuntimeError("operation changed frozen representation")
        atomic_json(project_path(config.output_dir) / "task_complete.json", report)
    elif stage in {"evaluation", "privileged_evaluation"}:
        privileged = stage == "privileged_evaluation"
        checkpoint = (str(project_path(config.output_dir) / "checkpoints/source_final.pt")
                      if privileged else selected_checkpoint(manifest, task["name"]))
        if privileged:
            trained = json.loads((project_path(config.output_dir) / "source_summary.json").read_text())
            if trained["step"] != config.train_steps:
                raise RuntimeError("privileged diagnostic did not reach matched source update count")
        scorer = selected_checkpoint(manifest, task["scorer"])
        report = evaluate(config, checkpoint, scorer, privileged=privileged)
        atomic_json(project_path(config.output_dir) / "task_complete.json", report)
    else:
        raise ValueError(stage)
    print(json.dumps({"event": "complete", "stage": stage, "index": index}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    execute(args.manifest, args.stage, args.index)


if __name__ == "__main__":
    main()
