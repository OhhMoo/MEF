"""Source-only calibration from frozen evidence state to the known flow scalar.

The successful generator, molecular encoder, evidence encoder H, response
model, and decoder stay fixed.  Only this small bridge is optimized.  The
bridge learns from source campaign coefficients and emits one scalar, which
is placed in response-state coordinate zero at deployment.
"""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .redesign_contracts import SupportBatch
from .redesign_sampling import frozen_deployment
from .redesign_train import cache_graph_features, tensor_fingerprint


SUPPORT_SIZES = (4, 16, 64)


class CalibrationBridge(nn.Module):
    """The deliberately small 64→64→1 evidence-to-coefficient map."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, frozen_state: Tensor) -> Tensor:
        if frozen_state.ndim != 2 or frozen_state.shape[1] != 64:
            raise ValueError("bridge input must be [episodes,64] frozen H states")
        return self.network(frozen_state.float())


def bridge_state(head: CalibrationBridge, frozen_state: Tensor) -> Tensor:
    """Insert inferred scalar into (s_hat,0,...), preserving state width."""
    estimate = head(frozen_state).squeeze(-1)
    result = torch.zeros_like(frozen_state)
    result[:, 0] = estimate
    return result


def _source_split_check(dataset, coefficients) -> tuple[str, ...]:
    if getattr(dataset, "split", None) != "train":
        raise ValueError("bridge training requires the source train split")
    source = tuple(dataset.campaign_ids)
    declared = tuple(dataset.manifest["splits"]["train"]["campaign_ids"])
    if source != declared or len(source) != len(set(source)):
        raise ValueError("source campaign identities differ from the manifest")
    other = set(dataset.manifest["splits"]["validation"]["campaign_ids"])
    other.update(dataset.manifest["splits"]["test"]["campaign_ids"])
    if set(source) & other:
        raise ValueError("source and held-out campaign identities overlap")
    if any(name not in coefficients or not np.isfinite(float(coefficients[name])) for name in source):
        raise ValueError("each source campaign needs one finite known coefficient")
    if not set(SUPPORT_SIZES).issubset(set(dataset.support_sizes)):
        raise ValueError("source dataset lacks a requested support size")
    return source


def _make_episodes(dataset, coefficients, rng, episodes_per_size):
    records = []
    source_count = len(dataset.campaign_ids)
    for repeat in range(episodes_per_size):
        for size_index, size in enumerate(SUPPORT_SIZES):
            campaign_index = (repeat * len(SUPPORT_SIZES) + size_index) % source_count
            episode = dataset.sample_episode(rng, support_size=size, queries_per_lead=1,
                                             campaign_index=campaign_index)
            records.append((episode.support_indices.copy(), episode.support_values.copy(),
                            episode.context.copy(), float(coefficients[episode.campaign_id]), size))
    return records


@torch.no_grad()
def _cache_frozen_states(model, features, records, device, batch_size=64):
    if features.shape[1] != model.redesign_config.feature_dim:
        raise ValueError("cached graph feature width differs from frozen evidence encoder")
    states = []
    for first in range(0, len(records), batch_size):
        batch = records[first:first + batch_size]
        slots = max(len(row[0]) for row in batch)
        indices = torch.zeros(len(batch), slots, dtype=torch.long)
        values = torch.zeros(len(batch), slots, 2, dtype=torch.float32)
        mask = torch.zeros(len(batch), slots, dtype=torch.bool)
        context = torch.zeros(len(batch), model.redesign_config.context_dim, dtype=torch.float32)
        for row_index, (member_indices, outcomes, row_context, _, _) in enumerate(batch):
            count = len(member_indices)
            indices[row_index, :count] = torch.as_tensor(member_indices, dtype=torch.long)
            values[row_index, :count] = torch.as_tensor(outcomes, dtype=torch.float32)
            context[row_index] = torch.as_tensor(row_context, dtype=torch.float32)
            mask[row_index, :count] = True
        selected = features.index_select(0, indices.flatten().to(device)).reshape(len(batch), slots, -1)
        support = SupportBatch(selected, values.to(device),
                               context.to(device), mask.to(device))
        states.append(model.support_state(support).float().cpu())
    result = torch.cat(states)
    if result.shape != (len(records), 64) or not torch.isfinite(result).all():
        raise ValueError("frozen H produced invalid bridge states")
    return result


def _fit_head(head: CalibrationBridge, states: Tensor, targets: Tensor, *,
              steps: int, seed: int, learning_rate: float = 1e-3) -> list[float]:
    """Optimize only the bridge on cached states; useful for linear-state tests."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if states.ndim != 2 or states.shape[1] != 64 or targets.shape != (len(states), 1):
        raise ValueError("states/targets must have shapes [episodes,64]/[episodes,1]")
    device = next(head.parameters()).device
    states, targets = states.to(device).float(), targets.to(device).float()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    head.train()
    history = []
    for step in range(steps):
        indices = torch.randint(len(states), (min(128, len(states)),), generator=rng)
        predicted = head(states.index_select(0, indices.to(device)))
        loss = (predicted - targets.index_select(0, indices.to(device))).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            history.append(float(loss.detach()))
    head.eval()
    return history


def _metrics(head, states, targets, sizes):
    device = next(head.parameters()).device
    with torch.no_grad():
        predicted = head(states.to(device)).squeeze(-1).cpu()
    target = targets.squeeze(-1).cpu()
    result = {"mae": float((predicted - target).abs().mean()),
              "mse": float((predicted - target).square().mean()),
              "sign_accuracy": float(((predicted >= 0) == (target >= 0)).float().mean())}
    for size in SUPPORT_SIZES:
        mask = torch.as_tensor([entry == size for entry in sizes], dtype=torch.bool)
        result[f"mae_n{size}"] = float((predicted[mask] - target[mask]).abs().mean())
        result[f"sign_accuracy_n{size}"] = float(
            ((predicted[mask] >= 0) == (target[mask] >= 0)).float().mean())
    return result


def train_bridge(model, dataset, coefficients, device, *, steps: int = 3000,
                 checkpoint_path, seed: int = 142, episodes_per_size: int = 1000,
                 validation_per_size: int = 64) -> tuple[CalibrationBridge, dict]:
    """Train a source-only head from frozen H states to known source s values.

    Episode query records are discarded.  Only measured support compounds,
    measured support outcomes, and source campaign coefficient targets reach
    the bridge training data.  The generator state fingerprint is checked after
    caching and optimization.
    """
    source = _source_split_check(dataset, coefficients)
    if getattr(model.redesign_config, "state_dim", None) != 64:
        raise ValueError("bridge requires a 64-dimensional frozen H state")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (episodes_per_size, validation_per_size)):
        raise ValueError("episode counts must be positive integers")
    device = torch.device(device)
    model_device = next(model.parameters()).device
    if (model_device.type != device.type
            or (device.index is not None and model_device.index != device.index)):
        raise ValueError("model must already reside on the requested device")
    device = model_device
    before = tensor_fingerprint(model.state_dict().items())
    rng = np.random.default_rng(seed)
    train_records = _make_episodes(dataset, coefficients, rng, episodes_per_size)
    validation_records = _make_episodes(dataset, coefficients, rng, validation_per_size)
    with frozen_deployment(model):
        features, graph_cache_seconds = cache_graph_features(
            model, dataset.graph_dataset, batch_size=256, bf16=False)
        features = features.to(device)
        train_states = _cache_frozen_states(model, features, train_records, device)
        validation_states = _cache_frozen_states(model, features, validation_records, device)
    targets = torch.as_tensor([row[3] for row in train_records], dtype=torch.float32)[:, None]
    validation_targets = torch.as_tensor([row[3] for row in validation_records],
                                          dtype=torch.float32)[:, None]
    # Fork RNG so bridge initialization does not change a runner's global RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = CalibrationBridge().to(device)
    history = _fit_head(head, train_states, targets, steps=steps, seed=seed)
    after = tensor_fingerprint(model.state_dict().items())
    if after != before:
        raise RuntimeError("frozen generator weights changed during bridge training")
    summary = {"format_version": 1, "source_split": dataset.split,
               "source_campaign_count": len(source), "source_campaign_ids": list(source),
               "training_episodes": len(train_records),
               "validation_episodes": len(validation_records),
               "support_sizes": list(SUPPORT_SIZES), "steps": steps, "seed": seed,
               "graph_cache_seconds": graph_cache_seconds,
               "generator_sha256_before": before, "generator_sha256_after": after,
               "training": _metrics(head, train_states, targets, [row[4] for row in train_records]),
               "validation": _metrics(head, validation_states, validation_targets,
                                      [row[4] for row in validation_records]),
               "loss_history": history}
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "head": head.state_dict(), "summary": summary}, path)
    head.requires_grad_(False).eval()
    return head, summary


def load_bridge(path, device) -> CalibrationBridge:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported calibration bridge checkpoint")
    head = CalibrationBridge().to(device)
    head.load_state_dict(payload["head"])
    return head.requires_grad_(False).eval()
