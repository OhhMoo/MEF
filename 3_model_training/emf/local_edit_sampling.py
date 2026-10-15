"""Prepare lead-anchored Gaussian starts for the standard NFE-counted sampler."""

import torch

from .contracts import GraphMasks, Latents
from .local_edit_model import LocalEditCondition


def anchored_initial_noise(gaussian: Latents, masks: GraphMasks,
                           condition: LocalEditCondition, sigma: float = 1.) -> Latents:
    """Return X0+sigma*epsilon in absolute coordinates for sample_redesign_latents.

    The caller supplies owned Gaussian draws, then passes this result as
    ``initial_noise`` to the existing sampler.  That sampler counts its
    terminal clean prediction in NFE and leaves one terminal decode.
    """
    if not torch.isfinite(torch.as_tensor(sigma)) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    lead = condition.aligned_lead(masks)
    if gaussian.nodes.shape != lead.nodes.shape or gaussian.pairs.shape != lead.pairs.shape:
        raise ValueError("Gaussian noise shape does not match aligned lead")
    return Latents(lead.nodes + sigma * gaussian.nodes,
                   lead.pairs + sigma * gaussian.pairs).masked(masks)
