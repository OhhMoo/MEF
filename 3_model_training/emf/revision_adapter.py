"""Source-distilled bounded state revision for a frozen conditional B generator.

This is a small graph-equivariant residual map, not a new flow model. Paired
same-noise old/new conditional rollouts provide teacher states at one fixed
solver time. Both contexts use observed-support ridge calibration; neither
private campaign coefficients nor response truth are read here. The fixed
final checkpoint uses hidden width 64, gain one and trust RMS .5. Validation
teacher-state error is diagnostic; molecular success is gated separately.

ELF transport remains in masked_edit_sampling; no upstream primitive is copied
here. Protected lead coordinates are restored exactly after the residual map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import math
import time

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import GraphMasks, Latents
from .masked_edit_data import DEFAULT_DIR, MaskedEditDataset, fit_support_response
from .masked_edit_model import MaskedEditCondition
from .masked_edit_revision import revision_grid
from .masked_edit_run import condition_for, load_model, now
from .masked_edit_sampling import make_paired_gaussian, sample_masked_edit_latents
from .redesign_train import atomic_json, digest, tensor_fingerprint


@dataclass(frozen=True)
class AdapterConfig:
    node_dim: int
    pair_dim: int
    context_dim: int = 4
    state_dim: int = 64
    hidden_dim: int = 64
    max_delta_rms: float = .5
    gain: float = 1.

    def __post_init__(self):
        if min(self.node_dim, self.pair_dim, self.context_dim,
               self.state_dim, self.hidden_dim) < 1:
            raise ValueError("adapter dimensions must be positive")
        if not math.isfinite(self.max_delta_rms) or self.max_delta_rms <= 0:
            raise ValueError("max_delta_rms must be finite and positive")
        if not math.isfinite(self.gain) or not 0 <= self.gain <= 1:
            raise ValueError("gain must be in [0,1]")


def _pool(value: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask[..., None], value.float(), 0.).sum(1) / mask.sum(1).clamp_min(1)[:, None]


def _editable_mean_square(delta: Latents, edit: GraphMasks) -> Tensor:
    nodes = torch.where(edit.nodes[..., None], delta.nodes.float(), 0.)
    pairs = torch.where(edit.pairs[..., None], delta.pairs.float(), 0.)
    count = (edit.nodes.sum(1) * delta.nodes.shape[-1]
             + edit.pairs.sum(1) * delta.pairs.shape[-1]).clamp_min(1)
    return (nodes.square().sum((1, 2)) + pairs.square().sum((1, 2))) / count


def editable_rms(delta: Latents, edit: GraphMasks) -> Tensor:
    """Joint RMS per graph, with finite derivative at the zero-initialized map."""
    squared = _editable_mean_square(delta, edit)
    return torch.where(squared > 0, squared.clamp_min(1e-24).sqrt(), 0.)


def state_error(actual: Latents, target: Latents, edit: GraphMasks) -> Tensor:
    return _editable_mean_square(Latents(actual.nodes - target.nodes,
                                         actual.pairs - target.pairs), edit)


class BoundedStateAdapter(nn.Module):
    """Shared per-site MLPs, invariant pooled chemistry and zero-initialized heads.

    The base generator is deliberately absent from this module's parameters.
    The trust bound is joint RMS over editable latent channels, per example.
    It is a state displacement bound, not a velocity or per-coordinate bound.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        task_dim = 2 * (config.context_dim + config.state_dim + 4)
        pooled_dim = 2 * (config.node_dim + config.pair_dim)
        common_dim = task_dim + pooled_dim + 3  # time and editable fractions

        def head(dim):
            net = nn.Sequential(nn.Linear(2 * dim + common_dim + 1, config.hidden_dim),
                                nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim),
                                nn.SiLU(), nn.Linear(config.hidden_dim, dim))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.node_map = head(config.node_dim)
        self.pair_map = head(config.pair_dim)
        self.register_buffer("selected_gain", torch.tensor(config.gain, dtype=torch.float32))
        self.register_buffer("max_delta_rms", torch.tensor(config.max_delta_rms, dtype=torch.float32))

    def _validate(self, state, old, new):
        old.validate(old.masks)
        new.validate(new.masks)
        if (not torch.equal(old.lead_graph.atoms, new.lead_graph.atoms)
                or not torch.equal(old.lead_graph.bonds, new.lead_graph.bonds)):
            raise ValueError("adapter revision must preserve aligned lead graph")
        for name in ("nodes", "pairs"):
            if (not torch.equal(getattr(old.masks, name), getattr(new.masks, name))
                    or not torch.equal(getattr(old.edit_masks, name), getattr(new.edit_masks, name))
                    or not torch.equal(getattr(old.lead, name), getattr(new.lead, name))):
                raise ValueError("adapter revision must preserve aligned lead and masks")
            value = getattr(state, name)
            width = self.config.node_dim if name == "nodes" else self.config.pair_dim
            if (value.shape != getattr(new.lead, name).shape or value.shape[-1] != width
                    or not torch.isfinite(value).all()):
                raise ValueError("state must be finite and match lead latent shapes")
        for condition in (old, new):
            batch = state.nodes.shape[0]
            for name, width in (("baseline", 2), ("goal", 2),
                                ("context", self.config.context_dim),
                                ("response_state", self.config.state_dim)):
                value = getattr(condition, name)
                if value.shape != (batch, width) or not torch.isfinite(value).all():
                    raise ValueError(f"invalid calibrated condition field: {name}")

    @staticmethod
    def _task(condition):
        return torch.cat([condition.baseline.float(), condition.context.float(),
                          condition.response_state.float(), condition.goal.float()], -1)

    def raw_delta(self, state: Latents, t: float | Tensor,
                  old_condition: MaskedEditCondition,
                  new_condition: MaskedEditCondition) -> Latents:
        self._validate(state, old_condition, new_condition)
        batch = state.nodes.shape[0]
        time_value = torch.as_tensor(t, device=state.nodes.device, dtype=torch.float32)
        if time_value.ndim == 0:
            time_value = time_value.expand(batch)
        if (time_value.shape != (batch,) or not torch.isfinite(time_value).all()
                or torch.any((time_value < 0) | (time_value >= 1))):
            raise ValueError("time must be finite scalar or [batch] in [0,1)")
        old, new = self._task(old_condition), self._task(new_condition)
        lead, masks, edit = new_condition.lead, new_condition.masks, new_condition.edit_masks
        # The function is exactly identity for unchanged deployment evidence/goals.
        changed = torch.any(old != new, dim=-1).float()[:, None, None]
        with torch.autocast(device_type=state.nodes.device.type, enabled=False):
            pooled = torch.cat([_pool(state.nodes, masks.nodes), _pool(state.pairs, masks.pairs),
                                _pool(lead.nodes, masks.nodes), _pool(lead.pairs, masks.pairs)], -1)
            common = torch.cat([old, new, pooled, time_value[:, None],
                                edit.nodes.float().mean(1)[:, None],
                                edit.pairs.float().sum(1)[:, None] / max(edit.pairs.shape[1], 1)], -1)

            def mapped(net, current, fixed, editable):
                inputs = torch.cat([current.float(), fixed.float(),
                                    common[:, None].expand(-1, current.shape[1], -1),
                                    editable[..., None].float()], -1)
                return torch.where(editable[..., None], net(inputs) * changed, 0.)

            return Latents(mapped(self.node_map, state.nodes, lead.nodes, edit.nodes),
                           mapped(self.pair_map, state.pairs, lead.pairs, edit.pairs))

    def apply_state(self, state: Latents, t: float | Tensor,
                    old_condition: MaskedEditCondition,
                    new_condition: MaskedEditCondition, *,
                    gain: float | None = None,
                    max_delta_rms: float | None = None) -> Latents:
        """Apply one bounded residual, restoring protected lead and padding exactly.

        Overrides support the explicit identity reference and focused checks.
        Deployment uses the fixed serialized values (omit both overrides).
        """
        gain_value = float(self.selected_gain) if gain is None else float(gain)
        bound = float(self.max_delta_rms) if max_delta_rms is None else float(max_delta_rms)
        if not math.isfinite(gain_value) or not 0 <= gain_value <= 1:
            raise ValueError("gain must be in [0,1]")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("max_delta_rms must be finite and positive")
        delta = self.raw_delta(state, t, old_condition, new_condition)
        rms = editable_rms(delta, new_condition.edit_masks)
        scale = (bound / rms.clamp_min(1e-12)).clamp(max=1.) * gain_value
        edit, valid, lead = new_condition.edit_masks, new_condition.masks, new_condition.lead

        def apply(current, residual, fixed, editable, active):
            changed = current + (scale[:, None, None] * residual).to(current.dtype)
            return torch.where(active[..., None],
                               torch.where(editable[..., None], changed, fixed), 0.)

        return Latents(apply(state.nodes, delta.nodes, lead.nodes, edit.nodes, valid.nodes),
                       apply(state.pairs, delta.pairs, lead.pairs, edit.pairs, valid.pairs))

    forward = apply_state


@dataclass(frozen=True)
class TeacherPair:
    old_state: Latents
    new_state: Latents
    old_condition: MaskedEditCondition
    new_condition: MaskedEditCondition
    time: float
    provenance: dict


def _condition(base, coefficients):
    state = base.response_state.clone()
    state[:, :6] = coefficients.reshape(1, 6).expand(state.shape[0], -1)
    return replace(base, response_state=state)


@torch.no_grad()
def _build_pairs(model, dataset, task, device):
    """Use only the loaded split's public observations, never the private oracle."""
    if dataset.split not in {"train", "validation"}:
        raise ValueError("adapter pairs may use source or validation only")
    nfe, T = int(task.get("nfe", 16)), float(task.get("T", .95))
    grid = revision_grid(nfe, T, (float(task.get("revision_time", .75)),))[0]
    index, actual = grid["grid_index"], grid["actual_time"]
    count = int(task.get("noises_per_campaign", 4))
    if count < 1:
        raise ValueError("noises_per_campaign must be positive")
    limit_key = "campaign_limit" if dataset.split == "train" else "validation_campaign_limit"
    limit = int(task.get(limit_key, len(dataset.campaign_ids)))
    if limit < 1:
        raise ValueError("campaign limits must be positive")
    scales = dataset.manifest["descriptor_scales_from_source_molecules"]
    records = []
    for ci, cid in enumerate(dataset.campaign_ids[:limit]):
        mask_id = dataset.manifest["campaigns"][cid]["mask_id"]
        mask = dataset.masks[mask_id]
        support = dataset.nested_support(ci, seed=77600 + ci, sizes=(4, 16))
        fits = {}
        for size in (4, 16):
            record = support["records"][size]
            # Source chemistry is mandatory for every source pair, with the
            # declared common lead as the same zero-reference exception as B.
            if dataset.split == "train" and not set(map(int, record["indices"])).issubset(
                    dataset.molecule_splits["train"] | {mask.lead_index}):
                raise ValueError("non-source support chemistry in adapter training")
            fits[size] = fit_support_response(
                [graph["smiles"] for graph in record["graphs"]], record["values"],
                mask.lead_smiles, scales, ridge=.05)
        base = condition_for(model, dataset, mask_id, count)
        noise_seed = (int(task["seed"]) * 100000 + ci
                      + (0 if dataset.split == "train" else 50000000))
        noise = make_paired_gaussian(model.config, count, noise_seed, device=device)
        scenarios = (("goal_reversal", "evidence_revision")
                     if task.get("include_evidence_revision", False) else ("goal_reversal",))
        for scenario in scenarios:
            old_size = 16 if scenario == "goal_reversal" else 4
            old_w = torch.as_tensor(fits[old_size]["coefficients"], device=device, dtype=torch.float32)
            new_w = torch.as_tensor(fits[16]["coefficients"], device=device, dtype=torch.float32).clone()
            if scenario == "goal_reversal":
                new_w[0] *= -1
            old, new = _condition(base, old_w), _condition(base, new_w)
            old_trace = sample_masked_edit_latents(model, old, noise, nfe=nfe, T=T, return_trace=True)
            new_trace = sample_masked_edit_latents(model, new, noise, nfe=nfe, T=T, return_trace=True)
            source, target = old_trace.states[index].detach(), new_trace.states[index].detach()
            distance = editable_rms(Latents(target.nodes - source.nodes,
                                          target.pairs - source.pairs), base.edit_masks)
            provenance = {"split": dataset.split, "campaign_id": cid, "mask_id": mask_id,
                          "scenario": scenario, "noise_seed": noise_seed, "count": count,
                          "grid_index": index, "actual_time": actual,
                          "target_editable_rms": distance.cpu().tolist(),
                          "support_indices": {str(size): support["records"][size]["indices"].tolist()
                                              for size in (4, 16)},
                          "calibration": {str(size): {"basis_rank": fits[size]["basis_rank"],
                                                       "ridge": .05} for size in (4, 16)}}
            records.append(TeacherPair(source, target, old, new, actual, provenance))
    if not records:
        raise ValueError("no adapter teacher pairs")
    return records


def _select(pair: TeacherPair, indices: Tensor) -> TeacherPair:
    def take(state):
        return Latents(state.nodes[indices], state.pairs[indices])
    return TeacherPair(take(pair.old_state), take(pair.new_state),
                       pair.old_condition.index_select(indices),
                       pair.new_condition.index_select(indices), pair.time, pair.provenance)


@torch.no_grad()
def _validation_mse(adapter, pairs, *, gain=1.):
    errors = []
    for pair in pairs:
        output = adapter.apply_state(pair.old_state, pair.time, pair.old_condition,
                                     pair.new_condition, gain=gain)
        errors.append(state_error(output, pair.new_state, pair.new_condition.edit_masks))
    return float(torch.cat(errors).mean())


def load_adapter(checkpoint, device="cpu", *, base_checkpoint=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("kind") != "bounded_revision_adapter":
        raise ValueError("not a supported revision adapter checkpoint")
    if payload.get("base_fingerprint_before") != payload.get("base_fingerprint_after"):
        raise ValueError("adapter checkpoint reports changed base parameters")
    if base_checkpoint is not None and digest(base_checkpoint) != payload["base_sha256"]:
        raise ValueError("adapter base checkpoint hash mismatch")
    adapter = BoundedStateAdapter(AdapterConfig(**payload["adapter_config"]))
    adapter.load_state_dict(payload["adapter_state"])
    adapter.to(device).eval().requires_grad_(False)
    return adapter, payload


def train(task: dict, device="cpu") -> dict:
    """Fit one fixed adapter; source gradients, validation diagnostics, no test access.

    Root's recovery pipeline owns the prospective recoverability gate and the
    separate exact-success validation check before any held-out test use.
    """
    required = ("checkpoint", "output", "seed")
    if any(name not in task for name in required):
        raise ValueError("adapter train requires checkpoint, output and seed")
    if task.get("split", "train") != "train":
        raise ValueError("adapter gradients require source train split")
    steps, batch_size = int(task.get("steps", 500)), int(task.get("batch_size", 64))
    interval = int(task.get("validation_every", 100))
    if min(steps, batch_size, interval) < 1:
        raise ValueError("steps, batch_size and validation_every must be positive")
    if (int(task.get("hidden_dim", 64)) != 64
            or float(task.get("max_delta_rms", .5)) != .5
            or float(task.get("gain", 1.)) != 1.
            or "gains" in task or "trust_bounds" in task):
        raise ValueError("fixed adapter requires hidden_dim=64, max_delta_rms=.5, gain=1; no sweep")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists():
        raise FileExistsError("adapter output already contains checkpoint.pt; choose a distinct output")
    started = time.perf_counter()
    seed = int(task["seed"])
    rng = torch.Generator(device="cpu").manual_seed(seed)
    model, base_payload = load_model(task["checkpoint"], device)
    if model.variant != "B" or not base_payload.get("conditioned", False):
        raise ValueError("adapter teacher requires a frozen task-conditioned B checkpoint")
    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    before = tensor_fingerprint(model.state_dict().items())
    base_sha = digest(task["checkpoint"])
    data_dir = Path(task.get("data_dir", DEFAULT_DIR))
    source = MaskedEditDataset(data_dir, "train", kind="vector")
    validation = MaskedEditDataset(data_dir, "validation", kind="vector")
    source_pairs = _build_pairs(model, source, task, device)
    validation_pairs = _build_pairs(model, validation, task, device)
    pair_seconds = time.perf_counter() - started
    config = AdapterConfig(model.config.node_dim, model.config.pair_dim,
                           model.redesign_config.context_dim, model.redesign_config.state_dim,
                           64, .5, 1.)
    # Avoid disturbing the caller's Torch RNG; paired solver noise has its own generator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        adapter = BoundedStateAdapter(config).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(task.get("lr", 1e-3)), weight_decay=0.)
    identity_mse = _validation_mse(adapter, validation_pairs, gain=0.)
    history = []
    for step in range(1, steps + 1):
        pair = source_pairs[int(torch.randint(len(source_pairs), (), generator=rng))]
        indices = torch.randint(pair.old_state.nodes.shape[0], (batch_size,), generator=rng)
        pair = _select(pair, indices.to(pair.old_state.nodes.device))
        optimizer.zero_grad(set_to_none=True)
        predicted = adapter.apply_state(pair.old_state, pair.time, pair.old_condition,
                                        pair.new_condition)
        loss = state_error(predicted, pair.new_state, pair.new_condition.edit_masks).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite adapter training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % interval == 0 or step == steps:
            row = {"step": step, "source_state_mse": float(loss.detach()),
                   "validation_state_mse": _validation_mse(adapter, validation_pairs),
                   "gain": 1., "max_delta_rms": .5}
            history.append(row)
    after = tensor_fingerprint(model.state_dict().items())
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("base model changed or accumulated gradients during adapter training")
    if digest(task["checkpoint"]) != base_sha:
        raise RuntimeError("base checkpoint file changed during adapter training")
    training_seconds = time.perf_counter() - started - pair_seconds
    final_metrics = history[-1]
    provenance = {"training_split": "train", "diagnostic_split": "validation", "test_access": False,
                  "source_campaigns": list(dict.fromkeys(pair.provenance["campaign_id"] for pair in source_pairs)),
                  "source_molecule_indices": sorted(source.molecule_splits["train"]),
                  "common_lead_exception": True, "data_manifest_sha256": digest(data_dir / "manifest.json"),
                  "train_npz_sha256": digest(data_dir / "train.npz"),
                  "validation_npz_sha256": digest(data_dir / "validation.npz"),
                  "calibration": "observed nested support N4/N16; descriptor ridge=0.05; no private W",
                  "base_training_task": base_payload.get("task", {}),
                  "initializer_exposure": "Inherited frozen B checkpoint representation may include prior broad reconstruction exposure."}
    pair_records = [pair.provenance for pair in source_pairs + validation_pairs]
    atomic_json(output / "pair_provenance.json", pair_records)
    payload = {"schema_version": 1, "kind": "bounded_revision_adapter",
               "adapter_config": asdict(config),
               "adapter_state": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
               "selected_gain": 1., "max_delta_rms": .5,
               "base_checkpoint": str(task["checkpoint"]), "base_sha256": base_sha,
               "base_fingerprint_before": before, "base_fingerprint_after": after,
               "selection": "fixed final checkpoint; exact decoded validation gate owned by recovery runner",
               "final_metrics": final_metrics, "identity_validation_state_mse": identity_mse,
               "training_seconds": training_seconds, "pair_generation_seconds": pair_seconds,
               "source_provenance": provenance, "task": task, "created_utc": now()}
    temporary = output / "checkpoint.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "checkpoint.pt")
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    distances = {split: [x for pair in pairs for x in pair.provenance["target_editable_rms"]]
                 for split, pairs in (("train", source_pairs), ("validation", validation_pairs))}
    summary = {"phase": "complete", "checkpoint": str(output / "checkpoint.pt"),
               "checkpoint_sha256": digest(output / "checkpoint.pt"), "steps": steps,
               "parameters": sum(p.numel() for p in adapter.parameters()),
               "source_pairs": sum(p.old_state.nodes.shape[0] for p in source_pairs),
               "validation_pairs": sum(p.old_state.nodes.shape[0] for p in validation_pairs),
               "teacher_forward_nfe": 2 * int(task.get("nfe", 16)) * sum(
                   p.old_state.nodes.shape[0] for p in source_pairs + validation_pairs),
               "pair_generation_seconds": pair_seconds, "elapsed_seconds": time.perf_counter() - started,
               "training_seconds": training_seconds,
               "selection": "fixed final checkpoint; gain=1; trust RMS=.5",
               "final_metrics": final_metrics,
               "identity_validation_state_mse": identity_mse,
               "teacher_target_distance": {split: {"median_rms": float(np.median(values)),
                   "fraction_outside_bounds": {str(bound): float(np.mean(np.asarray(values) > bound))
                                               for bound in (.5,)}} for split, values in distances.items()},
               "base_unchanged": before == after, "base_sha256": base_sha,
               "source_provenance": provenance, "created_utc": now(),
               "limitation": "Validation state MSE is a distillation diagnostic; exact molecular goal improvement requires separate validation. Paired teacher targets can lie outside the bounded local neighborhood."}
    atomic_json(output / "training_summary.json", summary)
    return summary
