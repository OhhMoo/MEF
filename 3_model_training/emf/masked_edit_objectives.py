"""Paired masked-edit objective with frozen contextual encoder targets."""

import math

import torch
from torch import Tensor

from .contracts import GraphBatch, GraphMasks, Latents
from .masked_edit_model import (MaskedEditCondition, MaskedEditModel,
                                hybrid_latents, validate_edit_masks)
from .objectives import categorical_loss, interpolate_latents
from .redesign import pool_latents
from .redesign_contracts import CampaignBatch
from .redesign_objectives import _draw, _noise, weighted_velocity_matching_loss


def _check_allowed_edits(lead: GraphBatch, query: GraphBatch, edit: GraphMasks) -> None:
    if (not torch.equal(lead.masks.nodes, query.masks.nodes)
            or not torch.equal(lead.masks.pairs, query.masks.pairs)):
        raise ValueError("lead and query require identical fixed-size alignment")
    validate_edit_masks(edit, lead.masks)
    if (torch.any((lead.atoms != query.atoms) & ~edit.nodes & lead.masks.nodes)
            or torch.any((lead.bonds != query.bonds) & ~edit.pairs & lead.masks.pairs)):
        raise ValueError("query changes a protected atom or bond category")


def masked_edit_loss(model: MaskedEditModel, batch: CampaignBatch,
                     edit_masks: GraphMasks, *, T: float = .95,
                     noise_scale: float = 1., decoder_sigma: float = .05,
                     response_weight: float = 1., decoder_weight: float = 1.,
                     accept_all: bool = False,
                     generator: torch.Generator | None = None,
                     flow_state_override: Tensor | None = None):
    """A uses full targets; B/C use protected-lead/editable-query hybrid targets.

    Masked variants apply flow error only to editable coordinates. Their
    decoder sees hybrid contextual latents and all valid graph labels, so it
    learns the protected context even when E(query) differs from E(lead).
    """
    if not 0 < T < 1 or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("require 0<T<1 and finite positive noise_scale")
    if not math.isfinite(decoder_sigma) or decoder_sigma < 0:
        raise ValueError("decoder_sigma must be finite and nonnegative")
    if any(not math.isfinite(x) or x < 0 for x in (response_weight, decoder_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    _check_allowed_edits(batch.lead, batch.query, edit_masks)
    with torch.no_grad():
        query = model.encode(batch.query)
        lead = model.encode(batch.lead)
        candidate_features = pool_latents(query, batch.query.masks, model.config.max_nodes)
        lead_features = pool_latents(lead, batch.lead.masks, model.config.max_nodes)
    need_evidence = response_weight > 0 or (flow_state_override is None and model.flow.task_conditioned)
    states = (model.support_state(batch.support).index_select(0, batch.episode_index)
              if need_evidence else batch.baseline.new_zeros(
                  (batch.query.atoms.shape[0], model.redesign_config.state_dim)))
    if flow_state_override is not None:
        if flow_state_override.shape != states.shape or not torch.isfinite(flow_state_override).all():
            raise ValueError("flow_state_override must be finite and match state shape")
    flow_state = states if flow_state_override is None else flow_state_override
    condition = MaskedEditCondition(lead_features, batch.baseline, batch.context,
                                    flow_state, batch.goal, lead, batch.lead, edit_masks)
    target = query if model.variant == "A" else hybrid_latents(lead, query, edit_masks)
    loss_masks = batch.query.masks if model.variant == "A" else edit_masks
    time = _draw((batch.query.atoms.shape[0],), query.nodes.device,
                 generator, normal=False) * T
    gaussian = _noise(query, loss_masks, generator)
    if model.variant == "B":
        start = hybrid_latents(lead, Latents(noise_scale * gaussian.nodes,
                                             noise_scale * gaussian.pairs), edit_masks)
    else:
        start = Latents(lead.nodes + noise_scale * gaussian.nodes,
                        lead.pairs + noise_scale * gaussian.pairs).masked(batch.query.masks)
    z = interpolate_latents(target, start, time, batch.query.masks)
    prediction = model.predict_clean(z, time, batch.query.masks, condition)
    if not accept_all and not torch.equal(batch.flow_weights.bool(),
                                           torch.all(batch.delta >= batch.goal, dim=-1)):
        raise ValueError("flow weights differ from measured feasibility")
    accepted = torch.ones_like(batch.flow_weights) if accept_all else batch.flow_weights.float()
    flow_loss = weighted_velocity_matching_loss(prediction, target, time,
                                                 loss_masks, accepted)
    if response_weight:
        distribution = model.predict_response_features(
            lead_features, candidate_features, batch.baseline, batch.context, states)
        with torch.autocast(device_type=query.nodes.device.type, enabled=False):
            response_loss = -distribution.log_prob(batch.delta.float()).mean()
    else:
        response_loss = flow_loss.new_zeros(())
    decoder_noise = _noise(target, loss_masks, generator)
    corrupted = Latents(target.nodes + decoder_sigma * decoder_noise.nodes,
                        target.pairs + decoder_sigma * decoder_noise.pairs).masked(batch.query.masks)
    decode_loss, parts = categorical_loss(model.decode(corrupted, batch.query.masks), batch.query)
    total = flow_loss + response_weight * response_loss + decoder_weight * decode_loss
    metrics = {"loss": total, "flow_loss": flow_loss, "response_loss": response_loss,
               "decode_loss": decode_loss, "accepted_fraction": accepted.mean(),
               "editable_atom_fraction": edit_masks.nodes.float().mean(),
               "editable_bond_fraction": edit_masks.pairs.float().mean(), **parts}
    return total, {key: value.detach() for key, value in metrics.items()}
