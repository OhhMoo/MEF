"""Pinned local-edit continuation: source training, blind evaluation, and diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import torch
import yaml

from .contracts import GraphMasks, Latents, ModelConfig
from .data import collate_graphs
from .models import EMFModel
from .redesign import RedesignModel
from .redesign_contracts import RedesignConfig
from .redesign_train import (RedesignRunConfig, atomic_json, digest, load_redesign_config,
    load_source_checkpoint, project_path, representation_fingerprint, run_training,
    tensor_fingerprint)
from .redesign_objectives import redesign_loss
from .redesign_sampling import frozen_deployment, SamplingPlan
from .steering_data import SteeringDataset
from .local_edit_data import AlignedSteeringDataset
from .local_edit_model import LocalEditModel
from .local_edit_objectives import local_edit_loss
from .local_edit_factorized import _NormalizedPooledSelector
from .local_edit_sampling import anchored_initial_noise

KINDS = ("pooled_known", "normalized_known", "anchored_known", "anchored_prior")


def task_path(checkpoint):
    return Path(checkpoint).parent.parent / "local_edit_task.json"


def architecture(config, task, base=None):
    if base is None:
        base = EMFModel(config.redesign.model)
    kind = task["kind"]
    if kind.startswith("anchored"):
        return LocalEditModel(config.redesign, base,
                              task_conditioned=kind != "anchored_prior")
    model = RedesignModel(config.redesign, base)
    if kind == "normalized_known":
        # Wrap the already initialized selector: no changed random draw or weight.
        model.flow.selector = _NormalizedPooledSelector(
            model.flow.selector, config.redesign.model.hidden_dim)
    return model


def load_checkpoint(path, device):
    path = project_path(path)
    if not task_path(path).exists():
        return load_source_checkpoint(path, device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(payload["config"])
    red = dict(values["redesign"])
    red["model"] = ModelConfig(**red["model"])
    values["redesign"] = RedesignConfig(**red)
    config = RedesignRunConfig(**values)
    task = json.loads(task_path(path).read_text())
    model = architecture(config, task).to(device)
    model.load_state_dict(payload["model"])
    if representation_fingerprint(model) != payload["representation_sha256"]:
        raise RuntimeError("frozen representation fingerprint differs")
    model.local_edit_task = task
    return model, config, payload


def _prepare_resume_checkpoint(task, config):
    """Amend only output/time limits in a copied, provenance-checked checkpoint."""
    if "resume_checkpoint" not in task:
        return None, None
    original_path = project_path(task["resume_checkpoint"]).resolve()
    if original_path.name != "source_final.pt" or not original_path.is_file():
        raise ValueError("source completion requires an existing partial source_final.pt")
    original_hash = digest(original_path)
    if task.get("resume_checkpoint_sha256") and task["resume_checkpoint_sha256"] != original_hash:
        raise ValueError("partial source checkpoint differs from completion decision")
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    old_config, new_config = original["config"], asdict(config)
    changed = {name for name in old_config.keys() | new_config.keys()
               if old_config.get(name) != new_config.get(name)}
    if changed != {"output_dir", "train_seconds"}:
        raise ValueError(f"completion may change only output_dir and train_seconds; changed={sorted(changed)}")
    old_output = project_path(old_config["output_dir"]).resolve()
    new_output = project_path(new_config["output_dir"]).resolve()
    if (original_path != old_output / "checkpoints/source_final.pt"
            or new_output == old_output):
        raise ValueError("completion must use a new output directory and the declared source checkpoint")
    old_step = original.get("step")
    if (isinstance(old_step, bool) or not isinstance(old_step, int)
            or not 0 < old_step < new_config["train_steps"]):
        raise ValueError("completion source must be partial and have an absolute update count")
    if (not new_config["train_seconds"] > old_config["train_seconds"]
            or not new_config["train_seconds"] > original["training_seconds"]):
        raise ValueError("completion needs a larger cumulative training-seconds cap")
    old_manifest_path = old_output / "manifest.json"
    old_task_path = old_output / "local_edit_task.json"
    old_config_path = old_output / "config.yaml"
    if not all(path.is_file() for path in (old_manifest_path, old_task_path, old_config_path)):
        raise ValueError("partial source provenance files are incomplete")
    old_manifest = json.loads(old_manifest_path.read_text())
    old_task = json.loads(old_task_path.read_text())
    if yaml.safe_load(old_config_path.read_text()) != old_config or old_manifest["config"] != old_config:
        raise ValueError("partial source config differs from its run records")
    for name in ("kind", "seed", "context_mode", "noise_scale", "operation", "parent_checkpoint"):
        if old_task.get(name) != task.get(name):
            raise ValueError(f"completion changed source task field {name}")
    data_hash = digest(project_path(config.data_dir) / "manifest.json")
    initializer_hash = digest(project_path(config.initializer))
    if (original["data_sha256"] != data_hash
            or old_manifest["data_sha256"] != data_hash
            or old_manifest["initializer_sha256"] != initializer_hash
            or old_manifest["representation_sha256"] != original["representation_sha256"]):
        raise ValueError("partial source data/initializer/representation provenance differs")
    # The loader strictly reconstructs the recorded architecture and verifies
    # the frozen encoder/normalizer fingerprint against the checkpoint.
    verified_model, _, _ = load_checkpoint(original_path, "cpu")
    model_hash = tensor_fingerprint(verified_model.state_dict().items())
    if model_hash != tensor_fingerprint(original["model"].items()):
        raise ValueError("partial model differs after strict architecture load")
    del verified_model
    new_output.mkdir(parents=True, exist_ok=True)
    destination = new_output / "resume_input.pt"
    if destination.exists() or (new_output / "checkpoints/latest.pt").exists():
        raise FileExistsError("completion output already contains a resume or training checkpoint")
    amendment = {"original_checkpoint": str(original_path), "original_sha256": original_hash,
                 "original_config": old_config, "original_step": old_step,
                 "original_training_seconds": original["training_seconds"],
                 "amended_config": new_config, "changed_fields": sorted(changed),
                 "data_sha256": data_hash,
                 "representation_sha256": original["representation_sha256"],
                 "model_sha256": model_hash}
    copied = dict(original)
    copied["config"] = new_config
    copied["resume_amendment"] = amendment
    temporary = destination.with_suffix(".tmp")
    torch.save(copied, temporary)
    temporary.replace(destination)
    atomic_json(new_output / "resume_provenance.json", {
        **amendment, "copied_resume_sha256": digest(destination)})
    return destination, amendment


def train(task, alignment_path):
    config = load_redesign_config(task["config"])
    output = project_path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "local_edit_task.json", task)
    resume_path, amendment = _prepare_resume_checkpoint(task, config)
    private = json.loads((project_path(config.data_dir) / "private/oracle.json").read_text())

    def factory(cfg, device):
        payload = torch.load(project_path(cfg.initializer), map_location="cpu", weights_only=False)
        if payload["config"]["model"] != asdict(cfg.redesign.model):
            raise ValueError("initializer architecture differs")
        base = EMFModel(cfg.redesign.model)
        base.load_state_dict(payload["model"])
        model = architecture(cfg, task, base).to(device)
        if task.get("parent_checkpoint"):
            parent = torch.load(task["parent_checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(parent["model"])
        model.local_edit_task = task
        return model, payload

    def dataset_factory(data_dir, split):
        if task["kind"].startswith("anchored"):
            return AlignedSteeringDataset(data_dir, split, alignment_path=alignment_path)
        return SteeringDataset(data_dir, split)

    def objective(model, batch, *, episodes, **kwargs):
        state = None
        if task.get("context_mode", "known") == "known":
            state = torch.zeros((len(episodes), model.redesign_config.state_dim),
                                device=batch.baseline.device)
            state[:, 0] = torch.tensor([private["coefficients"][e.campaign_id]
                                        for e in episodes], device=state.device)
            state = state.index_select(0, batch.episode_index)
        if task["kind"].startswith("anchored"):
            return local_edit_loss(model, batch, T=kwargs["T"],
                decoder_sigma=kwargs["sigma"], response_weight=kwargs["beta"],
                decoder_weight=kwargs["gamma"], generator=kwargs["generator"],
                flow_state_override=state, noise_scale=task.get("noise_scale", 1.),
                accept_all=task["kind"] == "anchored_prior",
                terminal_ce_weight=task.get("terminal_ce_weight", 0.))
        return redesign_loss(model, batch, flow_state_override=state, **kwargs)

    summary = run_training(config, dataset_class=dataset_factory, objective=objective,
        model_factory=factory, goal_rule="fixed .25/-0.10; original-query denominator",
        save_step_checkpoints=True, resume=resume_path)
    atomic_json(output / "parent_provenance.json", {
        "parent_checkpoint": task.get("parent_checkpoint"),
        "parent_sha256": digest(task["parent_checkpoint"]) if task.get("parent_checkpoint") else None,
        "resume_checkpoint": task.get("resume_checkpoint"),
        "resume_sha256": amendment["original_sha256"] if amendment else None,
        "source_updates_this_stage": summary["step"] - amendment["original_step"] if amendment else summary["step"],
        "cumulative_source_updates": summary["step"],
        "context_mode": task.get("context_mode", "known")})
    return summary


def _known_state(model, state, campaign_id, dataset):
    # Explicit known-context diagnostic, never used by inferred-context runs.
    private = json.loads((dataset.data_dir / "private/oracle.json").read_text())
    result = torch.zeros_like(state)
    result[:, 0] = private["coefficients"][campaign_id]
    return result


def _attach_control(model, surrogate_path, radius):
    from .local_edit_control import load_surrogate, correct_clean
    device = next(model.parameters()).device
    head = load_surrogate(surrogate_path, device=device)
    if isinstance(head, tuple):
        head = head[0]
    model.control_surrogate = head
    model.control_calls = {"flow_calls": 0, "surrogate_forward_calls": 0,
                           "surrogate_backward_calls": 0, "seconds": 0.,
                           "candidate_corrections": 0, "clipped_candidates": 0,
                           "fallback_candidates": 0, "remaining_deficit_sum": [0.,0.],
                           "call_count_unit": "batch calls; candidate_corrections counts per-attempt interventions"}
    original = model.predict_clean

    def predict(self, z, t, masks, condition):
        clean = original(z, t, masks, condition)
        started = time.perf_counter()
        corrected, stats = correct_clean(clean, masks, condition.lead_features,
            condition.baseline, condition.goal, condition.response_state[:, 0],
            self.control_surrogate, radius=radius)
        self.control_calls["flow_calls"] += 1
        self.control_calls["surrogate_forward_calls"] += 2
        self.control_calls["surrogate_backward_calls"] += 2
        self.control_calls["seconds"] += time.perf_counter() - started
        self.control_calls["candidate_corrections"] += len(stats["radius_clipped"])
        self.control_calls["clipped_candidates"] += sum(stats["radius_clipped"])
        self.control_calls["fallback_candidates"] += sum(not x for x in stats["linearized_feasible"])
        for i in range(2):
            self.control_calls["remaining_deficit_sum"][i] += sum(x[i] for x in stats["remaining_deficit_after"])
        self.control_last_stats = stats
        return corrected
    model.predict_clean = types.MethodType(predict, model)


def evaluate(task, manifest, split, *, mode=None, output_suffix=None, attempts=None):
    from .steering_evaluation import run_evaluation
    config = load_redesign_config(task["config"])
    mode = mode or task.get("context_mode", "known")
    ckpt = task.get("checkpoint") or str(project_path(config.output_dir) / "checkpoints/source_final.pt")
    operation = task.get("operation", False)
    config = replace(config, num_attempts=attempts or (16 if operation else
                     (128 if split == "validation" else 256)),
                     support_sizes=[4] if operation else ([64] if mode == "known" else [4,16,64]),
                     nfe=8 if operation else 32)
    radius = task.get("control_radius")
    surrogate_path = task.get("surrogate_checkpoint")
    def loader(path, device):
        model, cfg, payload = load_checkpoint(path, device)
        if str(Path(path).resolve()) == str(Path(ckpt).resolve()):
            if radius is not None:
                _attach_control(model, surrogate_path, radius)
            if mode == "bridge":
                from .local_edit_calibration import load_bridge
                model.coefficient_bridge = load_bridge(task["bridge_checkpoint"], device)
        return model, cfg, payload

    last_generator = []
    lead_templates = {}
    def prepare(generator, condition, plan, lead, campaign_id, dataset, scorer, scorer_condition):
        if isinstance(generator, LocalEditModel):
            # CUDA graph reductions may round differently on repeated calls.
            # Reuse the full lead encoding so an evidence intervention changes
            # neither its anchor nor its paired initial noise.
            key = (id(generator), tuple(lead.smiles))
            if key not in lead_templates:
                lead_templates[key] = generator.condition(
                    lead, condition.baseline, condition.context,
                    condition.response_state, condition.goal)
            prepared = replace(lead_templates[key],
                lead_features=condition.lead_features, baseline=condition.baseline,
                context=condition.context, response_state=condition.response_state,
                goal=condition.goal)
            expanded = prepared.expand(len(plan.counts))
            noise = Latents(plan.noise.nodes.to(lead.atoms.device),
                            plan.noise.pairs.to(lead.atoms.device))
            shifted = anchored_initial_noise(noise, expanded.lead_masks, expanded,
                                              sigma=task.get("noise_scale", 1.))
            plan = replace(plan, noise=Latents(shifted.nodes.cpu(), shifted.pairs.cpu()))
        else:
            prepared = condition
        if not last_generator:
            last_generator.append(generator)
        return prepared, plan

    callback = _known_state if mode == "known" else None
    support_callback = None
    if mode == "calibrated_control":
        def support_callback(model, state, support, campaign_id, dataset):
            from .local_edit_control import calibrate_coefficient
            lead_features = support.features[:, 0]
            coefficient, stats = calibrate_coefficient(model.control_surrogate,
                lead_features, support.features[0], support.values[0])
            result = torch.zeros_like(state)
            result[:, 0] = coefficient.reshape(-1)
            model.coefficient_fit = stats
            return result
    elif mode == "bridge":
        def support_callback(model, state, support, campaign_id, dataset):
            from .local_edit_calibration import bridge_state
            return bridge_state(model.coefficient_bridge, state)
    suffix = output_suffix or f"evaluation_{split}"
    out = project_path(config.output_dir) / suffix
    result = run_evaluation(config, ckpt, split, out,
        scorer_checkpoint=manifest["scorers"][str(config.seed)],
        model_loader=loader, flow_state_override=callback, proposal_setup=prepare,
        support_state_override=support_callback,
        protocol_extra={"study": "local_edit_001", "kind": task["kind"],
                        "context_mode": mode, "noise_scale": task.get("noise_scale", 1.),
                        "control_radius": radius, "parent_checkpoint": task.get("parent_checkpoint"),
                        "control_surrogate_sha256": digest(surrogate_path) if surrogate_path else None})
    if last_generator and hasattr(last_generator[0], "control_calls"):
        atomic_json(out / "control_cost.json", last_generator[0].control_calls)
    return result


def audit(task, manifest):
    from .local_edit_evaluation import reconstruction_audit
    from .local_edit_trace import trace_pair
    from .steering_evaluation import _campaign_groups, _support, _case_seed
    from .redesign_train import cache_graph_features
    from .redesign_sampling import make_sampling_plan
    config = load_redesign_config(task["config"])
    checkpoint = project_path(config.output_dir) / "checkpoints/source_final.pt"
    model, _, _ = load_checkpoint(checkpoint, config.device)
    data = AlignedSteeringDataset(config.data_dir, "validation",
                                 alignment_path=manifest["alignment_path"])
    output = project_path(config.output_dir) / "diagnostics"
    output.mkdir(exist_ok=True)
    reconstruction_audit(model, data, config.device, output_path=output / "reconstruction.json")
    features, _ = cache_graph_features(model, data.graph_dataset, bf16=False)
    _, pairs = _campaign_groups(data)
    records = []
    with frozen_deployment(model):
        for pair in (pairs[0], next(p for p in pairs if p["family"] != pairs[0]["family"])):
            c1, c2 = pair["campaigns"]
            nested = data.nested_support(c1, pair["lead_index"],
                seed=data.paired_support_seed(data.campaign_ids[c1]), sizes=(64,))
            lead = collate_graphs([nested["lead"]]).to(config.device)
            states = []
            for campaign in (c1, c2):
                other = data.nested_support(campaign, pair["lead_index"],
                    seed=data.paired_support_seed(data.campaign_ids[campaign]), sizes=(64,))
                support = _support(other["records"][64], features, config.device)
                state = model.support_state(support)
                if task.get("context_mode", "known") == "known":
                    state = _known_state(model, state, data.campaign_ids[campaign], data)
                states.append(state)
            baseline = torch.zeros((1,2), device=config.device)
            context = torch.tensor(data.context, device=config.device)[None]
            goal = torch.tensor([[.25,-.10]], device=config.device)
            shared = model.condition(lead,baseline,context,states[0],goal)
            conditions = [replace(shared,response_state=s).expand(4) for s in states]
            plan = make_sampling_plan(model.config, data.count_prior,9,4,
                                      count_seed=998+pair["family"],noise_seed=999+pair["family"])
            noise = Latents(plan.noise.nodes.to(config.device),plan.noise.pairs.to(config.device))
            if isinstance(model, LocalEditModel):
                noise = anchored_initial_noise(noise,conditions[0].lead_masks,conditions[0],
                                               sigma=task.get("noise_scale",1.))
            records.append({"family":pair["family"],"trace":trace_pair(model, plan.counts,
                conditions[0],conditions[1],noise,nfe=16,include_canonical=True,
                known_state_jvp=task.get("context_mode", "known") == "known")})
    atomic_json(output / "trace.json",records)


def control(task, manifest):
    from .local_edit_control import train_surrogate
    config = load_redesign_config(task["config"])
    checkpoint = project_path(config.output_dir) / "checkpoints/source_final.pt"
    model, _, _ = load_checkpoint(checkpoint, config.device)
    data = SteeringDataset(config.data_dir,"train")
    private = json.loads((data.data_dir/"private/oracle.json").read_text())
    path = project_path(config.output_dir) / "control_surrogate.pt"
    with frozen_deployment(model), torch.enable_grad():
        # The new head is trainable; source model stays frozen.
        _, summary = train_surrogate(model, data, private["coefficients"],
            device=config.device, steps=3000, checkpoint_path=path, seed=config.seed)
    atomic_json(path.with_suffix(".json"), summary)
    for radius in (.05,.15):
        controlled = {**task,"surrogate_checkpoint":str(path),"control_radius":radius}
        evaluate(controlled, manifest, "validation",mode="known",
                 output_suffix=f"control_{radius:.2f}_validation")


def calibrate(task, manifest, *, test=False):
    config = load_redesign_config(task["config"])
    output = project_path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "local_edit_task.json", task)
    if task["context_mode"] == "bridge":
        from .local_edit_calibration import train_bridge
        model, _, _ = load_checkpoint(task["checkpoint"], config.device)
        model.requires_grad_(False).eval()
        source = SteeringDataset(config.data_dir,"train")
        private = json.loads((source.data_dir/"private/oracle.json").read_text())
        _, summary = train_bridge(model, source, private["coefficients"], config.device,
            steps=3000, checkpoint_path=task["bridge_checkpoint"], seed=config.seed)
    else:
        summary = {"chemical_weights_unchanged":True,
                   "calibration":"least squares from supplied support observations only",
                   "source_surrogate_checkpoint":task["surrogate_checkpoint"]}
    atomic_json(output/"calibration_summary.json",summary)
    evaluate(task,manifest,"validation")
    if test:
        evaluate(task,manifest,"test")
    atomic_json(output/"task_complete.json",summary)


def execute(manifest_path, stage, index):
    manifest = json.loads(Path(manifest_path).read_text())
    for path, expected in {**manifest["snapshot_sha256"],**manifest["input_sha256"]}.items():
        if digest(path) != expected:
            raise RuntimeError(f"pinned input changed: {path}")
    if stage == "source_completion":
        decision_path=Path(manifest["study_root"])/"source_completion_decision.json"
        decision=json.loads(decision_path.read_text())
        tasks=decision["tasks"]
        for task in tasks:
            if not task.get("resume_checkpoint") or not task.get("resume_checkpoint_sha256"):
                raise ValueError("source-completion task requires a pinned partial checkpoint")
            expected=decision["resolved_config_sha256"].get(task["config"])
            if expected is None or digest(task["config"]) != expected:
                raise RuntimeError("resolved source-completion configuration changed")
    elif stage == "control":
        completion_path=Path(manifest["study_root"])/"source_completion_decision.json"
        if completion_path.exists():
            completed=json.loads(completion_path.read_text())
            tasks=[task for task in completed["effective_source"]
                   if task["kind"] == "anchored_prior"]
            if len(tasks) != 2 or {task["seed"] for task in tasks} != {142,143}:
                raise ValueError("control stage requires both effective anchored-prior source seeds")
            for task in tasks:
                expected=completed["resolved_config_sha256"].get(task["config"])
                if task.get("resume_checkpoint") and expected is None:
                    raise RuntimeError("effective completion task lacks a pinned configuration")
                if expected is not None and digest(task["config"]) != expected:
                    raise RuntimeError("effective source-completion configuration changed")
        else:
            tasks=manifest["stages"][stage]["tasks"]
    elif stage in {"continuation","calibration","final","control_final"}:
        decision_path=Path(manifest["study_root"])/"continuation_decision.json"
        decision=json.loads(decision_path.read_text())
        tasks=decision.get(stage,[])
        for path, expected in decision.get("resolved_config_sha256",{}).items():
            if digest(path) != expected:
                raise RuntimeError(f"resolved continuation configuration changed: {path}")
    else:
        tasks=manifest["stages"][stage]["tasks"]
    if index >= len(tasks):
        raise IndexError("task index outside declared stage")
    task=tasks[index]
    print(json.dumps({"event":"start","stage":stage,"index":index,"task":task}),flush=True)
    if stage == "calibration" or (stage == "continuation" and task.get("continuation_kind") == "calibration"):
        calibrate(task,manifest,test=stage=="calibration")
    elif stage in {"source","source_completion","operation","continuation"}:
        summary=train(task,manifest["alignment_path"])
        config=load_redesign_config(task["config"])
        audit(task,manifest)
        evaluate(task,manifest,"validation")
        if stage == "operation":
            subprocess.run([sys.executable,
                str(Path(manifest["snapshot"])/"review/review_local_edit_operation.py"),
                "--manifest",str(manifest_path)],check=True)
        atomic_json(project_path(config.output_dir)/"task_complete.json",summary)
    elif stage=="control":
        control(task,manifest)
    elif stage=="final":
        evaluate(task,manifest,"test")
    elif stage=="control_final":
        config=load_redesign_config(task["config"])
        for radius in (.05,.15):
            controlled={**task,"surrogate_checkpoint":str(project_path(config.output_dir)/"control_surrogate.pt"),
                        "control_radius":radius}
            evaluate(controlled,manifest,"test",mode="known",
                     output_suffix=f"control_{radius:.2f}_test")
    else:
        raise ValueError(stage)
    print(json.dumps({"event":"complete","stage":stage,"index":index}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--stage",required=True)
    p.add_argument("--index",type=int,default=int(os.environ.get("SLURM_ARRAY_TASK_ID","0")))
    a=p.parse_args()
    execute(a.manifest,a.stage,a.index)

if __name__=="__main__":
    main()
