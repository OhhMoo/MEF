"""Source-only clean-latent response surrogate and privileged local controller.

The first endpoint is represented as a unit-coefficient response ``du``;
deployment multiplies it by the separately supplied campaign coefficient. The
coefficient is privileged diagnostic context and must not be used in the ordinary
measured-evidence generator. No decoder, chemistry tool, or exact oracle is used
inside the correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import GraphMasks, Latents
from .data import collate_graphs
from .redesign import pool_latents


class CleanResponseSurrogate(nn.Module):
    """Predict source-normalized response from frozen pooled graph latents."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(nn.Linear(3 * feature_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 2))

    def forward(self, lead_features: Tensor, candidate_features: Tensor) -> Tensor:
        if lead_features.shape != candidate_features.shape or lead_features.shape[-1] != self.feature_dim:
            raise ValueError("lead/candidate features must share surrogate feature dimension")
        baseline = self.net(torch.cat((lead_features, lead_features,
                                       torch.zeros_like(lead_features)), dim=-1))
        candidate = self.net(torch.cat((lead_features, candidate_features,
                                        candidate_features - lead_features), dim=-1))
        return candidate - baseline


def _source_training_rows(dataset, coefficients) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset.split != "train" or any(not name.startswith("train_") for name in dataset.campaign_ids):
        raise ValueError("surrogate fitting requires source campaigns only")
    if hasattr(coefficients, "coefficients"):
        coefficients = coefficients.coefficients
    leads, queries, labels = [], [], []
    for campaign, name in enumerate(dataset.campaign_ids):
        coefficient = float(coefficients[name])
        if not np.isfinite(coefficient) or coefficient == 0:
            raise ValueError("source coefficient must be finite and nonzero")
        family = dataset.campaign_family[name]
        indices = np.asarray(dataset.family_indices[family], dtype=np.int64)
        lead = int(dataset.family_leads[family])
        target = dataset.observations[campaign, indices].astype(np.float32).copy()
        target[:, 0] /= coefficient
        if not np.isfinite(target).all():
            raise ValueError("source response target is nonfinite")
        leads.extend([lead] * len(indices))
        queries.extend(indices.tolist())
        labels.append(target)
    return (np.asarray(leads, dtype=np.int64), np.asarray(queries, dtype=np.int64),
            np.concatenate(labels, axis=0))


def train_surrogate(model, dataset, coefficients, device, *, steps: int = 3000,
                    batch_size: int = 256, learning_rate: float = 1e-3,
                    hidden_dim: int = 128, seed: int = 142,
                    checkpoint_path=None) -> tuple[CleanResponseSurrogate, dict]:
    """Fit all source-family records from frozen molecular features.

    ``coefficients`` is the private source split only. Its use makes this a
    separately labelled privileged diagnostic; held-out campaign labels never
    enter this fit. The trained checkpoint contains the head and fit summary.
    """
    if min(steps, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("invalid surrogate training budget")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    features = []
    with torch.no_grad():
        for offset in range(0, len(dataset), 128):
            records = [dataset.graph(index) for index in range(offset, min(offset + 128, len(dataset)))]
            features.append(model.graph_features(collate_graphs(records).to(device)).detach())
    model.train(was_training)
    features = torch.cat(features, dim=0).to(device)
    lead_indices, query_indices, targets = _source_training_rows(dataset, coefficients)
    lead_indices = torch.as_tensor(lead_indices, device=device)
    query_indices = torch.as_tensor(query_indices, device=device)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        surrogate = CleanResponseSurrogate(features.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for step in range(steps):
        selected = torch.randint(len(targets), (batch_size,), device=device, generator=generator)
        prediction = surrogate(features[lead_indices[selected]], features[query_indices[selected]])
        loss = (prediction - targets[selected]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 10) == 0:
            losses.append({"step": step + 1, "batch_mse": float(loss.detach().cpu())})
    surrogate.eval().requires_grad_(False)
    with torch.no_grad():
        predicted = surrogate(features[lead_indices], features[query_indices])
        errors = predicted - targets
        rmse = errors.square().mean(0).sqrt().cpu().tolist()
        lead_error = surrogate(features[lead_indices], features[lead_indices]).abs().max().item()
    summary = {"source_records": len(targets), "source_campaigns": len(dataset.campaign_ids),
               "source_molecules": len(dataset), "steps": steps, "batch_size": batch_size,
               "source_rmse_du_dy2": rmse, "lead_zero_max_abs": lead_error,
               "loss_trace": losses, "privileged_source_coefficient_fit": True}
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feature_dim": surrogate.feature_dim, "hidden_dim": hidden_dim,
                    "state_dict": surrogate.state_dict(), "summary": summary}, path)
    return surrogate, summary


def load_surrogate(path, device) -> tuple[CleanResponseSurrogate, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CleanResponseSurrogate(payload["feature_dim"], payload["hidden_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval().requires_grad_(False), payload["summary"]


@torch.no_grad()
def calibrate_coefficient(surrogate: CleanResponseSurrogate, lead_features: Tensor,
                          support_features: Tensor, support_values: Tensor, *,
                          ridge: float = 1e-6) -> tuple[Tensor, dict]:
    """Fit the first-endpoint campaign scale from measured support only.

    The first support row must be the lead. The head supplies source-fitted
    unit-coefficient changes ``du``; a scalar ridge least-squares fit relates
    them to observed first-endpoint changes from the lead. No oracle coefficient
    or withheld query response is read.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if lead_features.ndim != 2 or lead_features.shape[0] != 1:
        raise ValueError("lead_features must be [1,feature_dim]")
    if support_features.ndim != 2 or support_features.shape[1] != lead_features.shape[1]:
        raise ValueError("support_features must be [support,feature_dim]")
    size = support_features.shape[0]
    if size < 2 or support_values.shape != (size, 2):
        raise ValueError("support_values must be [support,2] with lead and one analogue")
    if not torch.allclose(support_features[0], lead_features[0], atol=1e-5, rtol=1e-5):
        raise ValueError("first support feature must be the lead")
    if not (torch.isfinite(lead_features).all() and torch.isfinite(support_features).all()
            and torch.isfinite(support_values).all()):
        raise ValueError("support calibration inputs must be finite")
    lead = lead_features.expand(size, -1)
    du = surrogate(lead, support_features)[:, 0]
    observed = support_values[:, 0] - support_values[0, 0]
    denominator = du.square().sum()
    coefficient = (du * observed).sum() / (denominator + ridge)
    residual = coefficient * du - observed
    summary = {"support_size": size, "informative_support": bool(denominator > ridge),
               "unit_response_energy": float(denominator.cpu()),
               "coefficient_estimate": float(coefficient.cpu()),
               "first_endpoint_support_rmse": float(residual.square().mean().sqrt().cpu()),
               "uses_support_measurements_only": True}
    return coefficient, summary


def _minimum_norm_two_constraints(jacobian: Tensor, deficit: Tensor,
                                  ridge: float) -> tuple[Tensor, Tensor]:
    """Project zero onto J delta >= signed gaps by the 0/1/2 active sets.

    A ridge-stabilized least-squares direction is returned only if linearized
    constraints have no numerically feasible candidate; callers can identify
    this from the returned solution mask. Negative gaps retain available margin.
    """
    batch, endpoints, dimensions = jacobian.shape
    if endpoints != 2 or deficit.shape != (batch, 2):
        raise ValueError("two endpoint Jacobian and deficit required")
    gram = jacobian @ jacobian.transpose(1, 2)
    tolerance = 1e-6
    directions = [torch.zeros((batch, dimensions), device=jacobian.device,
                              dtype=jacobian.dtype)]
    for endpoint in range(2):
        norm_squared = gram[:, endpoint, endpoint].clamp_min(1e-20)
        multiplier = deficit[:, endpoint] / norm_squared
        directions.append(jacobian[:, endpoint] * multiplier[:, None])
    determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1].square()
    scale = (gram[:, 0, 0] * gram[:, 1, 1]).clamp_min(0)
    safe = (scale > 0) & (determinant > 16 * torch.finfo(gram.dtype).eps * scale)
    inverse_determinant = torch.where(safe, determinant, torch.ones_like(determinant)).reciprocal()
    inverse_determinant = torch.where(safe, inverse_determinant,
                                      torch.zeros_like(inverse_determinant))
    lambda0 = (gram[:, 1, 1] * deficit[:, 0] - gram[:, 0, 1] * deficit[:, 1]) * inverse_determinant
    lambda1 = (gram[:, 0, 0] * deficit[:, 1] - gram[:, 0, 1] * deficit[:, 0]) * inverse_determinant
    directions.append(jacobian[:, 0] * lambda0[:, None] + jacobian[:, 1] * lambda1[:, None])
    candidates = torch.stack(directions, dim=1)
    achieved = torch.einsum("bkd,bcd->bkc", candidates, jacobian)
    feasible = (achieved >= deficit[:, None, :] - tolerance).all(-1)
    feasible[:, 3] &= safe
    norms = candidates.square().sum(-1)
    choice = torch.where(feasible, norms, torch.full_like(norms, float("inf"))).argmin(-1)
    any_feasible = feasible.any(-1)
    selected = candidates[torch.arange(batch, device=jacobian.device), choice]
    damped = torch.linalg.solve(gram + ridge * torch.eye(2, device=gram.device,
                                                       dtype=gram.dtype)[None], deficit)
    fallback = torch.einsum("bcd,bc->bd", jacobian, damped)
    return torch.where(any_feasible[:, None], selected, fallback), any_feasible


def correct_clean(clean: Latents, masks: GraphMasks, lead_features: Tensor,
                  baseline: Tensor, goal: Tensor, coefficient: Tensor | float,
                  surrogate: CleanResponseSurrogate, *, radius: float = 0.05,
                  ridge: float = 1e-4) -> tuple[Latents, dict]:
    """One bounded minimum-norm response correction at a clean prediction.

    Jacobians are taken only through the frozen surrogate and invariant latent
    pooling. The two linearized inequalities use signed gaps, preserving any
    margin already achieved. Clipping can reduce the repair; stats show both.
    """
    if radius <= 0 or ridge <= 0:
        raise ValueError("radius and ridge must be positive")
    batch = clean.nodes.shape[0]
    device = clean.nodes.device
    coefficient = torch.as_tensor(coefficient, dtype=clean.nodes.dtype, device=device).reshape(-1)
    if len(coefficient) == 1:
        coefficient = coefficient.expand(batch)
    if coefficient.shape != (batch,) or not torch.isfinite(coefficient).all():
        raise ValueError("coefficient must be one finite value per candidate")
    if baseline.shape != (batch, 2) or goal.shape != (batch, 2):
        raise ValueError("baseline and goal must be [batch,2]")
    with torch.enable_grad():
        nodes = clean.nodes.detach().float().requires_grad_(True)
        pairs = clean.pairs.detach().float().requires_grad_(True)
        candidate = pool_latents(Latents(nodes, pairs), masks, masks.nodes.shape[1])
        unit = surrogate(lead_features.detach().float(), candidate)
        predicted = torch.stack((unit[:, 0] * coefficient.float(), unit[:, 1]), dim=-1)
        thresholds = torch.stack((goal[:, 0], goal[:, 1] - baseline[:, 1]), dim=-1)
        deficit = thresholds - predicted
        gradients = []
        for endpoint in range(2):
            node_gradient, pair_gradient = torch.autograd.grad(
                predicted[:, endpoint].sum(), (nodes, pairs),
                retain_graph=endpoint == 0, allow_unused=True)
            if node_gradient is None:
                node_gradient = torch.zeros_like(nodes)
            if pair_gradient is None:
                pair_gradient = torch.zeros_like(pairs)
            gradients.append(torch.cat((node_gradient.mul(masks.nodes[..., None]).flatten(1),
                                        pair_gradient.mul(masks.pairs[..., None]).flatten(1)), dim=1))
        jacobian = torch.stack(gradients, dim=1)
        delta, linearized_feasible = _minimum_norm_two_constraints(jacobian, deficit, ridge)
        active_coordinates = (masks.nodes.sum(1) * nodes.shape[-1]
                              + masks.pairs.sum(1) * pairs.shape[-1]).clamp_min(1)
        rms = (delta.square().sum(1) / active_coordinates).sqrt()
        scale = (radius / rms.clamp_min(1e-20)).clamp(max=1)
        delta = delta * scale[:, None]
        node_count = nodes[0].numel()
        corrected = Latents((nodes + delta[:, :node_count].reshape_as(nodes)).detach(),
                            (pairs + delta[:, node_count:].reshape_as(pairs)).detach()).masked(masks)
        with torch.no_grad():
            corrected_features = pool_latents(corrected, masks, masks.nodes.shape[1])
            after_unit = surrogate(lead_features.detach().float(), corrected_features)
            predicted_after = torch.stack((after_unit[:, 0] * coefficient.float(),
                                           after_unit[:, 1]), dim=-1)
    stats = {"predicted_before": predicted.detach().cpu().tolist(),
             "predicted_after": predicted_after.detach().cpu().tolist(),
             "signed_gap_before": deficit.detach().cpu().tolist(),
             "signed_gap_after": (thresholds - predicted_after).detach().cpu().tolist(),
             "remaining_deficit_after": (thresholds - predicted_after).clamp_min(0).detach().cpu().tolist(),
             "linearized_feasible": linearized_feasible.detach().cpu().tolist(),
             "post_clip_surrogate_threshold_met":
                 (predicted_after >= thresholds).all(-1).detach().cpu().tolist(),
             "projection_method": ["active_set" if feasible else "damped_fallback"
                                   for feasible in linearized_feasible.detach().cpu().tolist()],
             "latent_rms_before_clip": rms.detach().cpu().tolist(),
             "radius_clipped": (scale < 1).detach().cpu().tolist(),
             "surrogate_forward_calls": 2, "surrogate_backward_calls": 2}
    return corrected, stats
