"""Feasibility-weighted flow, all-query response, and terminal graph decoding."""

import math

import torch
from torch import Tensor

from .contracts import GraphMasks, Latents
from .objectives import categorical_loss, interpolate_latents
from .redesign import RedesignModel, pool_latents
from .redesign_contracts import CampaignBatch, FlowCondition


def goal_satisfied(delta: Tensor, baseline: Tensor, goals: Tensor) -> Tensor:
    """Goals are supplied independently; outcomes determine this loss weight."""
    return ((delta.float()[..., 0] >= goals.float()[..., 0])
            & (baseline.float()[..., 1] + delta.float()[..., 1] >= goals.float()[..., 1]))


def weighted_velocity_matching_loss(predicted: Latents, clean: Latents, t: Tensor,
                                    masks: GraphMasks, accepted: Tensor) -> Tensor:
    """Mean I_g * squared velocity error over the original query batch.

    Each molecule divides by its active latent-coordinate count and (1-t)^2.
    Rejected queries remain in the batch denominator. No accepted queries gives
    a differentiable zero flow loss, while the other objectives still learn.
    """
    if torch.any((t < 0) | (t >= 1)):
        raise ValueError("velocity matching requires 0 <= t < 1")
    if accepted.shape != t.shape or torch.any((accepted != 0) & (accepted != 1)):
        raise ValueError("accepted must supply one zero/one feasibility weight per query")
    denominator = (1 - t.float())[:, None, None]
    node_error = ((predicted.nodes.float() - clean.nodes.float()) / denominator).square()
    pair_error = ((predicted.pairs.float() - clean.pairs.float()) / denominator).square()
    error = (torch.where(masks.nodes[..., None], node_error, 0).sum((1, 2))
             + torch.where(masks.pairs[..., None], pair_error, 0).sum((1, 2)))
    coordinates = (masks.nodes.sum(1) * clean.nodes.shape[-1]
                   + masks.pairs.sum(1) * clean.pairs.shape[-1])
    return (accepted.float() * error / coordinates.clamp_min(1)).mean()


def _draw(shape, device, generator, *, normal=True):
    draw_device = device if generator is None else generator.device
    function = torch.randn if normal else torch.rand
    return function(shape, device=draw_device, generator=generator, dtype=torch.float32).to(device)


def _noise(clean: Latents, masks: GraphMasks, generator) -> Latents:
    return Latents(_draw(clean.nodes.shape, clean.nodes.device, generator),
                   _draw(clean.pairs.shape, clean.pairs.device, generator)).masked(masks)


def redesign_loss(model: RedesignModel, batch: CampaignBatch, T=.95, sigma=.05,
                  beta=1., gamma=1., generator: torch.Generator | None = None,
                  flow_state_override: Tensor | None = None):
    """Three source losses; query outcomes enter weights and response labels only.

    An explicit generator isolates time/interpolation/decoder-noise draws from
    initialization so factorized and ordinary conditional runs can be paired.
    Goals are already drawn by the episode builder without query outcomes.
    """
    if not 0 < T < 1 or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("require 0<T<1 and finite sigma>=0")
    if any(not math.isfinite(weight) or weight <= 0 for weight in (beta, gamma)):
        raise ValueError("response and decoder weights must be finite and positive")
    measured_acceptance = goal_satisfied(batch.delta, batch.baseline, batch.goal)
    if not torch.equal(batch.flow_weights.bool(), measured_acceptance):
        raise ValueError("flow weights differ from measured joint-profile feasibility")
    with torch.no_grad():
        encoded = model.encode(batch.query)
        clean = Latents(encoded.nodes.float(), encoded.pairs.float())
        lead = model.graph_features(batch.lead)
        candidate = pool_latents(clean, batch.query.masks, model.config.max_nodes)
    states = model.support_state(batch.support).index_select(0, batch.episode_index)
    if flow_state_override is not None:
        if flow_state_override.shape != states.shape or not torch.isfinite(flow_state_override).all():
            raise ValueError("flow-state override must be finite with one state per query")
    flow_states = states if flow_state_override is None else flow_state_override
    condition = FlowCondition(lead, batch.baseline, batch.context, flow_states, batch.goal)
    time = _draw((len(batch.query.atoms),), clean.nodes.device, generator, normal=False) * T
    interpolation_noise = _noise(clean, batch.query.masks, generator)
    noisy = interpolate_latents(clean, interpolation_noise, time, batch.query.masks)
    predicted = model.predict_clean(noisy, time, batch.query.masks, condition)
    flow_loss = weighted_velocity_matching_loss(predicted, clean, time, batch.query.masks, batch.flow_weights)
    distribution = model.predict_response_features(lead, candidate, batch.baseline, batch.context, states)
    with torch.autocast(device_type=clean.nodes.device.type, enabled=False):
        response_loss = -distribution.log_prob(batch.delta.float()).mean()
    decoder_noise = _noise(clean, batch.query.masks, generator)
    corrupted = Latents(clean.nodes + sigma * decoder_noise.nodes,
                        clean.pairs + sigma * decoder_noise.pairs).masked(batch.query.masks)
    decode_loss, parts = categorical_loss(model.decode(corrupted, batch.query.masks), batch.query)
    loss = flow_loss + beta * response_loss + gamma * decode_loss
    covariance = distribution.covariance_matrix.detach()
    scales = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
    correlation = covariance[:, 0, 1] / (scales[:, 0] * scales[:, 1])
    metrics = {"loss": loss, "flow_loss": flow_loss, "response_loss": response_loss,
               "decode_loss": decode_loss, **parts,
               "accepted_fraction": batch.flow_weights.float().mean(),
               "accepted_queries": batch.flow_weights.float().sum(),
               "response_scale_1": scales[:, 0].mean(), "response_scale_2": scales[:, 1].mean(),
               "response_correlation": correlation.mean()}
    return loss, {key: value.detach() for key, value in metrics.items()}
