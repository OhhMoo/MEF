"""Continuous ELF-style destination transport, followed by an endpoint prediction.

No atom or bond categories are chosen here. ``nfe`` includes the final clean
prediction: nfe - 1 Euler calls integrate to T, then the last call predicts X.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import GraphMasks, Latents


@dataclass(frozen=True)
class SamplingResult:
    latents: Latents
    masks: GraphMasks
    counts: Tensor
    nfe: int


def model_device(model) -> torch.device:
    """Find the device without requiring a parameterized test oracle."""
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _normal(shape, device, generator) -> Tensor:
    # A supplied CPU generator also supports reproducible sampling on CUDA.
    draw_device = device if generator is None else generator.device
    return torch.randn(shape, generator=generator, device=draw_device,
                       dtype=torch.float32).to(device)


@torch.no_grad()
def sample_latents(model, counts: Tensor, *, nfe: int = 32, T: float = 0.95,
                   generator: torch.Generator | None = None,
                   initial_noise: Latents | None = None) -> SamplingResult:
    """Generate masked continuous destinations with exactly ``nfe`` flow calls.

    ``counts`` fixes each graph's number of active atoms for its entire path.
    Supplying initial_noise makes solver experiments possible without changing
    the ordinary Gaussian sampling path. It is still masked before transport.
    """
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 2:
        raise ValueError("nfe must be an integer >= 2, including the endpoint call")
    if not 0.0 < T < 1.0:
        raise ValueError("T must satisfy 0 < T < 1")

    config = model.config
    device = model_device(model)
    counts = torch.as_tensor(counts, device=device)
    if counts.is_floating_point() and not torch.equal(counts, counts.round()):
        raise ValueError("atom counts must be integers")
    counts = counts.long()
    masks = GraphMasks.from_counts(counts, config.max_nodes)
    batch = counts.numel()
    node_shape = (batch, config.max_nodes, config.node_dim)
    pair_shape = (batch, config.max_nodes * (config.max_nodes - 1) // 2,
                  config.pair_dim)
    if initial_noise is None:
        z = Latents(_normal(node_shape, device, generator),
                    _normal(pair_shape, device, generator))
    else:
        if initial_noise.nodes.shape != node_shape or initial_noise.pairs.shape != pair_shape:
            raise ValueError("initial_noise shape does not match model dimensions/counts")
        z = Latents(initial_noise.nodes.to(device=device, dtype=torch.float32),
                    initial_noise.pairs.to(device=device, dtype=torch.float32))
    z = z.masked(masks)

    # [0,T] excludes the singular t=1 endpoint, so the denominator is exact.
    times = torch.linspace(0.0, T, nfe, device=device, dtype=torch.float32)
    for step in range(nfe - 1):
        t = times[step].expand(batch)
        predicted = model.flow(z, t, masks).masked(masks)
        dt = times[step + 1] - times[step]
        denom = 1.0 - times[step]
        z = Latents(z.nodes + dt * (predicted.nodes - z.nodes) / denom,
                    z.pairs + dt * (predicted.pairs - z.pairs) / denom).masked(masks)

    endpoint = model.flow(z, times[-1].expand(batch), masks).masked(masks)
    return SamplingResult(endpoint, masks, counts, nfe)
