"""The EMF math written directly as small, masked PyTorch objectives.

ELF's clean-x prediction is trained through velocity error. For t<1,
||v_pred-v_target||² = ||x_pred-x||²/(1-t)². E is detached during flow
training; at lambda_mid=0, categorical supervision reaches q alone.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import GraphBatch, GraphLogits, GraphMasks, Latents
from .models import EMFModel


def _noise_like(latents: Latents, masks: GraphMasks) -> Latents:
    return Latents(torch.randn_like(latents.nodes), torch.randn_like(latents.pairs)).masked(masks)


def interpolate_latents(clean: Latents, noise: Latents, t: Tensor, masks: GraphMasks) -> Latents:
    """Z_t=(1-t)epsilon+tX on active atom/unordered-pair coordinates."""
    weight = t.float()[:, None, None]
    return Latents((1 - weight) * noise.nodes + weight * clean.nodes,
                   (1 - weight) * noise.pairs + weight * clean.pairs).masked(masks)


def velocity_matching_loss(predicted: Latents, clean: Latents, t: Tensor,
                           masks: GraphMasks) -> Tensor:
    """Mean per-molecule velocity error, normalized by its active coordinates."""
    if torch.any((t < 0) | (t >= 1)):
        raise ValueError("velocity matching requires 0 <= t < 1")
    denominator = (1 - t.float())[:, None, None]
    node_error = ((predicted.nodes.float() - clean.nodes.float()) / denominator).square()
    pair_error = ((predicted.pairs.float() - clean.pairs.float()) / denominator).square()
    error = (torch.where(masks.nodes[..., None], node_error, 0).sum(dim=(1, 2))
             + torch.where(masks.pairs[..., None], pair_error, 0).sum(dim=(1, 2)))
    coordinates = (masks.nodes.sum(dim=1) * clean.nodes.shape[-1]
                   + masks.pairs.sum(dim=1) * clean.pairs.shape[-1])
    return (error / coordinates.clamp_min(1)).mean()


def _categorical_mean(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if mask.shape[1] == 0:
        return logits.float().sum() * 0
    targets = torch.where(mask, targets, 0)
    loss = F.cross_entropy(logits.float().transpose(1, 2), targets.long(), reduction="none")
    per_molecule = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_molecule.mean()


def categorical_loss(logits: GraphLogits, batch: GraphBatch) -> tuple[Tensor, dict[str, Tensor]]:
    """Average atoms and active unordered pairs separately, then add them."""
    atom_ce = _categorical_mean(logits.atoms, batch.atoms, batch.masks.nodes)
    bond_ce = _categorical_mean(logits.bonds, batch.bonds, batch.masks.pairs)
    return atom_ce + bond_ce, {"atom_ce": atom_ce, "bond_ce": bond_ce}


def _corrupt(clean: Latents, masks: GraphMasks, sigma: float) -> Latents:
    if sigma == 0:
        return clean
    noise = _noise_like(clean, masks)
    return Latents(clean.nodes + sigma * noise.nodes,
                   clean.pairs + sigma * noise.pairs).masked(masks)


def reconstruction_loss(model: EMFModel, batch: GraphBatch, corruption: float = 0):
    """Warm-start E and q with graph reconstruction, optionally under corruption."""
    if corruption < 0:
        raise ValueError("corruption must be nonnegative")
    clean = model.encode(batch)
    loss, components = categorical_loss(model.decode(_corrupt(clean, batch.masks, corruption),
                                                    batch.masks), batch)
    return loss, {name: value.detach() for name, value in {"loss": loss, **components}.items()}


def flow_decoder_loss(model: EMFModel, batch: GraphBatch, T: float = .95,
                      sigma: float = .05, lambda_mid: float = 0):
    """L_flow + L_end + lambda_mid L_mid; the first exploration uses lambda=0."""
    if not 0 < T < 1:
        raise ValueError("T must lie strictly between zero and one")
    if sigma < 0 or lambda_mid < 0:
        raise ValueError("sigma and lambda_mid must be nonnegative")
    with torch.no_grad():
        encoded = model.encode(batch)
        clean = Latents(encoded.nodes.float(), encoded.pairs.float())
    t = torch.rand(batch.atoms.shape[0], device=batch.atoms.device) * T
    noisy = interpolate_latents(clean, _noise_like(clean, batch.masks), t, batch.masks)
    predicted = model.flow(noisy, t, batch.masks)
    flow_loss = velocity_matching_loss(predicted, clean, t, batch.masks)
    end_loss, components = categorical_loss(model.decode(_corrupt(clean, batch.masks, sigma),
                                                        batch.masks), batch)
    mid_loss = flow_loss.new_zeros(())
    if lambda_mid:
        mid_loss, _ = categorical_loss(model.decode(predicted, batch.masks), batch)
    loss = flow_loss + end_loss + lambda_mid * mid_loss
    metrics = {"loss": loss, "flow_loss": flow_loss, "end_loss": end_loss,
               "mid_loss": mid_loss, **components}
    return loss, {name: value.detach() for name, value in metrics.items()}
