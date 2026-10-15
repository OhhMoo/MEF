"""Explicitly conditioned continuous redesign with one terminal graph decode.

The Euler schedule follows the ELF-derived sampling study, retaining its
continuous destination-to-velocity conversion and terminal clean prediction.
This module imports the existing graph adaptation and leaves upstream unchanged.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from .contracts import GraphMasks, Latents, ModelConfig
from .sampling import SamplingResult, _normal, model_device


@dataclass(frozen=True)
class SamplingPlan:
    """Owned CPU counts and Gaussian draws, reused across evidence interventions."""

    counts: Tensor
    noise: Latents
    count_seed: int
    noise_seed: int
    lead_count: int
    creation_seconds: float = 0.0


def make_sampling_plan(config: ModelConfig, count_prior, lead_count: int,
                       num_attempts: int = 1000, *, count_seed: int = 42,
                       noise_seed: int = 43) -> SamplingPlan:
    """Draw from a source-pair conditional count row before any flow calls.

    Rows and columns index counts 1..max_nodes. Unsupported lead rows fail
    explicitly; an unconditional fallback would change the declared proposal law.
    Separate owned generators make draws independent of batching and support.
    """
    started = perf_counter()
    if isinstance(num_attempts, bool) or not isinstance(num_attempts, int) or num_attempts < 1:
        raise ValueError("num_attempts must be a positive integer")
    if isinstance(lead_count, bool) or not isinstance(lead_count, int) or not 1 <= lead_count <= config.max_nodes:
        raise ValueError("lead_count must be in 1..max_nodes")
    prior = torch.as_tensor(count_prior, dtype=torch.float64, device="cpu")
    if (prior.shape != (config.max_nodes, config.max_nodes)
            or not torch.isfinite(prior).all() or (prior < 0).any()):
        raise ValueError("count_prior must be a finite nonnegative [max_nodes,max_nodes] matrix")
    row = prior[lead_count - 1]
    if row.sum() <= 0:
        raise ValueError(f"source pairs do not support lead atom count {lead_count}")
    row = row / row.sum()
    count_rng = torch.Generator(device="cpu").manual_seed(count_seed)
    noise_rng = torch.Generator(device="cpu").manual_seed(noise_seed)
    counts = torch.multinomial(row, num_attempts, replacement=True,
                              generator=count_rng).long() + 1
    noise = Latents(torch.randn(num_attempts, config.max_nodes, config.node_dim,
                                generator=noise_rng),
                    torch.randn(num_attempts, config.max_nodes * (config.max_nodes - 1) // 2,
                                config.pair_dim, generator=noise_rng))
    return SamplingPlan(counts, noise, count_seed, noise_seed, lead_count,
                        perf_counter() - started)


@contextmanager
def frozen_deployment(model):
    """Hold every parameter fixed and disable stochastic training layers.

    Restore flags and individual module modes afterward so the same source
    training runner can inspect validation campaigns without changing its state.
    Gradients and tensor values are not modified.
    """
    parameters = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    for parameter, _ in parameters:
        parameter.requires_grad_(False)
    try:
        with torch.no_grad():
            yield model
    finally:
        for parameter, flag in parameters:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


@torch.no_grad()
def sample_redesign_latents(model, counts: Tensor, condition, *, nfe: int = 32,
                            T: float = 0.95,
                            generator: torch.Generator | None = None,
                            initial_noise: Latents | None = None,
                            base_only: bool = False) -> SamplingResult:
    """Run nfe-1 Euler calls plus one terminal clean-prediction call.

    Campaign conditioning is an explicit argument to every flow call. Counts
    and masks remain fixed; no graph categories are sampled during transport.
    """
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 2:
        raise ValueError("nfe must be an integer >= 2 including the terminal prediction")
    if not 0.0 < T < 1.0:
        raise ValueError("T must satisfy 0 < T < 1")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("deployment parameters must be frozen; use frozen_deployment")
    if any(module.training for module in model.modules()):
        raise ValueError("deployment modules must be in evaluation mode")
    if base_only and not hasattr(model.flow, "decompose"):
        raise ValueError("base_only requires the factorized model's explicit F/B decomposition")
    device, config = model_device(model), model.config
    counts = torch.as_tensor(counts, device=device)
    if counts.is_floating_point() and not torch.equal(counts, counts.round()):
        raise ValueError("atom counts must be integers")
    counts = counts.long()
    masks = GraphMasks.from_counts(counts, config.max_nodes)
    batch = counts.numel()
    node_shape = (batch, config.max_nodes, config.node_dim)
    pair_shape = (batch, config.max_nodes * (config.max_nodes - 1) // 2, config.pair_dim)
    if initial_noise is None:
        z = Latents(_normal(node_shape, device, generator),
                    _normal(pair_shape, device, generator))
    else:
        if initial_noise.nodes.shape != node_shape or initial_noise.pairs.shape != pair_shape:
            raise ValueError("initial_noise does not match candidate counts/model dimensions")
        z = Latents(initial_noise.nodes.to(device=device, dtype=torch.float32),
                    initial_noise.pairs.to(device=device, dtype=torch.float32))
    z = z.masked(masks)
    times = torch.linspace(0.0, T, nfe, device=device, dtype=torch.float32)

    def predict(state, time):
        if base_only:
            return model.flow.decompose(state, time, masks, condition).base.masked(masks)
        return model.predict_clean(state, time, masks, condition).masked(masks)

    for step in range(nfe - 1):
        prediction = predict(z, times[step].expand(batch))
        dt = times[step + 1] - times[step]
        denominator = 1.0 - times[step]
        z = Latents(z.nodes + dt * (prediction.nodes - z.nodes) / denominator,
                    z.pairs + dt * (prediction.pairs - z.pairs) / denominator).masked(masks)
    clean = predict(z, times[-1].expand(batch))
    return SamplingResult(clean, masks, counts, nfe)
