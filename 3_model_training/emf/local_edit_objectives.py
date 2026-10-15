"""Velocity matching around an aligned lead, with a terminal graph decoder."""

import math

import torch
from torch import Tensor

from .contracts import GraphBatch, GraphMasks, Latents
from .objectives import categorical_loss, interpolate_latents
from .redesign import pool_latents
from .redesign_contracts import CampaignBatch
from .redesign_objectives import (_draw, _noise, goal_satisfied,
                                  weighted_velocity_matching_loss)
from .local_edit_model import LocalEditCondition, LocalEditModel


def local_edit_loss(model: LocalEditModel, batch: CampaignBatch,
                    *, T: float = .95, noise_scale: float = 1., decoder_sigma: float = .05,
                    response_weight: float = 1., decoder_weight: float = 1.,
                    terminal_ce_weight: float = 0., accept_all: bool = False,
                    generator: torch.Generator | None = None,
                    flow_state_override: Tensor | None = None):
    """Train on aligned pairs with optional known campaign state in the flow.

    The flow sees Z_t=(1-t)(X0+noise_scale*epsilon)+t*X1 and predicts X1.
    ``accept_all`` trains a task-zero prior on every local pair; the
    conditional run uses the measured goal-feasibility weights.  Query
    outcomes only determine those weights and the response likelihood.
    """
    if not 0 < T < 1 or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("require 0<T<1 and positive finite noise_scale")
    if not math.isfinite(decoder_sigma) or decoder_sigma < 0:
        raise ValueError("decoder_sigma must be finite and nonnegative")
    if any(not math.isfinite(w) or w < 0 for w in
           (response_weight, decoder_weight, terminal_ce_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    queries = len(batch.query.atoms)
    with torch.no_grad():
        clean = model.encode(batch.query)
        lead = model.encode(batch.lead)
        candidate_features = pool_latents(clean, batch.query.masks, model.config.max_nodes)
        lead_features = pool_latents(lead, batch.lead.masks, model.config.max_nodes)
    states = model.support_state(batch.support).index_select(0, batch.episode_index)
    if flow_state_override is not None:
        if flow_state_override.shape != states.shape or not torch.isfinite(flow_state_override).all():
            raise ValueError("flow_state_override must be finite and [queries,state_dim]")
    if not accept_all and not torch.equal(
            batch.flow_weights.bool(), goal_satisfied(batch.delta, batch.baseline, batch.goal)):
        raise ValueError("flow weights differ from measured joint-profile feasibility")
    flow_states = states if flow_state_override is None else flow_state_override
    condition = LocalEditCondition(lead_features, batch.baseline, batch.context,
                                   flow_states, batch.goal, lead, batch.lead.masks)
    condition.aligned_lead(batch.query.masks)
    t = _draw((queries,), clean.nodes.device, generator, normal=False) * T
    noise = _noise(clean, batch.query.masks, generator)
    anchored_noise = Latents(lead.nodes + noise_scale * noise.nodes,
                             lead.pairs + noise_scale * noise.pairs).masked(batch.query.masks)
    z = interpolate_latents(clean, anchored_noise, t, batch.query.masks)
    prediction = model.predict_clean(z, t, batch.query.masks, condition)
    accepted = (torch.ones_like(batch.flow_weights) if accept_all
                else batch.flow_weights.float())
    flow_loss = weighted_velocity_matching_loss(prediction, clean, t,
                                                batch.query.masks, accepted)
    distribution = model.predict_response_features(
        lead_features, candidate_features, batch.baseline,
        batch.context, states)
    with torch.autocast(device_type=clean.nodes.device.type, enabled=False):
        response_loss = -distribution.log_prob(batch.delta.float()).mean()
    decoder_noise = _noise(clean, batch.query.masks, generator)
    corrupted = Latents(clean.nodes + decoder_sigma * decoder_noise.nodes,
                        clean.pairs + decoder_sigma * decoder_noise.pairs).masked(batch.query.masks)
    decode_loss, parts = categorical_loss(model.decode(corrupted, batch.query.masks), batch.query)
    terminal_loss = flow_loss.new_zeros(())
    if terminal_ce_weight:
        active = accepted.bool()
        terminal_targets = GraphBatch(
            batch.query.atoms, batch.query.bonds,
            GraphMasks(batch.query.masks.nodes & active[:, None],
                       batch.query.masks.pairs & active[:, None]))
        # Rejected rows contribute zero while the original query batch stays
        # in the denominator, matching goal-filtered velocity supervision.
        terminal_loss, _ = categorical_loss(
            model.decode(prediction, batch.query.masks), terminal_targets)
    total = (flow_loss + response_weight * response_loss + decoder_weight * decode_loss
             + terminal_ce_weight * terminal_loss)
    metrics = {"loss": total, "flow_loss": flow_loss, "response_loss": response_loss,
               "decode_loss": decode_loss, "terminal_ce_loss": terminal_loss,
               "accepted_fraction": accepted.mean(), **parts}
    return total, {key: value.detach() for key, value in metrics.items()}
