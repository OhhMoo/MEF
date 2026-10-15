"""Differentiable, NFE-counted masked-edit rollout and terminal graph decode."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import GraphLogits, Latents, ModelConfig
from .masked_edit_model import MaskedEditCondition, MaskedEditModel, decode_masked_edit
from .sampling import SamplingResult


@dataclass(frozen=True)
class MaskedSamplingResult(SamplingResult):
    """Optional states are Z at every evaluation time, including terminal T."""

    states: tuple[Latents, ...] = ()
    times: Tensor | None = None


def make_paired_gaussian(config: ModelConfig, batch_size: int, seed: int,
                         *, device: str | torch.device = "cpu") -> Latents:
    """One owned CPU draw can be cached and reused exactly across A/B/C."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    count = config.max_nodes
    return Latents(torch.randn(batch_size, count, config.node_dim, generator=rng),
                   torch.randn(batch_size, count * (count - 1) // 2,
                               config.pair_dim, generator=rng)).to(device)


def _clamp(state: Latents, condition: MaskedEditCondition) -> Latents:
    edit = condition.edit_masks
    lead = condition.lead
    return Latents(torch.where(edit.nodes[..., None], state.nodes, lead.nodes),
                   torch.where(edit.pairs[..., None], state.pairs, lead.pairs)).masked(condition.masks)


def _project_control(control: Latents, condition: MaskedEditCondition) -> Latents:
    if (control.nodes.shape != condition.lead.nodes.shape
            or control.pairs.shape != condition.lead.pairs.shape):
        raise ValueError("velocity control shape must match lead latents")
    return control.masked(condition.edit_masks)


def masked_initial_state(variant: str, gaussian: Latents,
                         condition: MaskedEditCondition,
                         noise_scale: float = 1.) -> Latents:
    """A/C begin near lead; B begins at zero-centered noise on editable sites."""
    if not torch.isfinite(torch.as_tensor(noise_scale)) or noise_scale <= 0:
        raise ValueError("noise_scale must be finite and positive")
    lead, masks = condition.lead, condition.masks
    if gaussian.nodes.shape != lead.nodes.shape or gaussian.pairs.shape != lead.pairs.shape:
        raise ValueError("Gaussian shapes must match lead latent shapes")
    if variant == "B":
        state = Latents(noise_scale * gaussian.nodes, noise_scale * gaussian.pairs)
        return _clamp(state, condition)
    if variant in {"A", "C"}:
        state = Latents(lead.nodes + noise_scale * gaussian.nodes,
                        lead.pairs + noise_scale * gaussian.pairs).masked(masks)
        return state if variant == "A" else _clamp(state, condition)
    raise ValueError("variant must be A, B, or C")


def sample_masked_edit_latents(
        model: MaskedEditModel, condition: MaskedEditCondition,
        gaussian: Latents | None = None, *, nfe: int = 32, T: float = .95,
        noise_scale: float = 1., start_t: float = 0.,
        initial_state: Latents | None = None,
        velocity_controller: Callable[[Latents, Tensor, MaskedEditCondition], Latents] | None = None,
        condition_schedule: Callable[[Latents, Tensor, MaskedEditCondition], MaskedEditCondition] | None = None,
        truncate_steps: int | None = None,
        return_trace: bool = False,
        controller: Callable[[Latents, Tensor, MaskedEditCondition], MaskedEditCondition] | None = None,
        ) -> MaskedSamplingResult:
    """Euler transport with editable control and one terminal clean prediction.

    The controller adds u(t) to the base velocity at every evaluation. Its
    terminal contribution is (1-T)u(T), consistent with clean destination =
    state + (1-t)velocity. A condition schedule may vary task fields while
    retaining the same lead and mask. A supplied intermediate state and
    start_t continue a prefix without drawing new noise. All steps preserve
    autograd; truncate_steps detaches before the last k evaluations.
    """
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 2:
        raise ValueError("nfe must be an integer >=2, including terminal prediction")
    if not 0 <= start_t < T < 1:
        raise ValueError("require 0<=start_t<T<1")
    if (initial_state is None) != (start_t == 0):
        raise ValueError("initial_state is required exactly when start_t>0")
    if initial_state is None and gaussian is None:
        raise ValueError("supply paired Gaussian noise or a continuation state")
    if truncate_steps is not None and (isinstance(truncate_steps, bool)
                                       or not isinstance(truncate_steps, int)
                                       or not 1 <= truncate_steps <= nfe):
        raise ValueError("truncate_steps must be in 1..nfe")
    if controller is not None:
        if condition_schedule is not None:
            raise ValueError("supply one condition schedule")
        condition_schedule = controller
    masks = condition.masks
    condition.validate(masks)
    batch = masks.nodes.shape[0]
    if initial_state is None:
        if gaussian.nodes.device != condition.lead.nodes.device or gaussian.pairs.device != condition.lead.pairs.device:
            gaussian = gaussian.to(condition.lead.nodes.device)
        z = masked_initial_state(model.variant, gaussian, condition, noise_scale)
    else:
        if (initial_state.nodes.shape != condition.lead.nodes.shape
                or initial_state.pairs.shape != condition.lead.pairs.shape):
            raise ValueError("continuation state shape must match lead latents")
        z = initial_state.to(condition.lead.nodes.device).masked(masks)
        if model.variant != "A":
            z = _clamp(z, condition)
    times = torch.linspace(start_t, T, nfe, device=z.nodes.device, dtype=torch.float32)
    states = []

    def evaluation(state: Latents, time: Tensor) -> tuple[Latents, Latents]:
        supplied = (condition if condition_schedule is None
                    else condition_schedule(state, time, condition))
        if not isinstance(supplied, MaskedEditCondition):
            raise TypeError("condition_schedule must return MaskedEditCondition")
        if (supplied.lead is not condition.lead
                or supplied.edit_masks is not condition.edit_masks
                or supplied.lead_graph is not condition.lead_graph):
            raise ValueError("condition schedule must preserve lead and editable masks")
        predicted = model.predict_clean(state, time, masks, supplied).masked(masks)
        control = (Latents(torch.zeros_like(state.nodes), torch.zeros_like(state.pairs))
                   if velocity_controller is None
                   else _project_control(velocity_controller(state, time, supplied), condition))
        return predicted, control

    for step in range(nfe - 1):
        if truncate_steps is not None and step == nfe - truncate_steps:
            z = z.detach()
        if return_trace:
            states.append(z)
        time = times[step]
        predicted, control = evaluation(z, time.expand(batch))
        dt = times[step + 1] - time
        denom = 1 - time
        base = Latents((predicted.nodes - z.nodes) / denom,
                       (predicted.pairs - z.pairs) / denom)
        if model.variant != "A":
            base = base.masked(condition.edit_masks)
        z = Latents(z.nodes + dt * (base.nodes + control.nodes),
                    z.pairs + dt * (base.pairs + control.pairs)).masked(masks)
        if model.variant != "A":
            z = _clamp(z, condition)
    if truncate_steps == 1:
        z = z.detach()
    if return_trace:
        states.append(z)
    predicted, control = evaluation(z, times[-1].expand(batch))
    clean = Latents(predicted.nodes + (1 - times[-1]) * control.nodes,
                    predicted.pairs + (1 - times[-1]) * control.pairs).masked(masks)
    if model.variant != "A":
        clean = _clamp(clean, condition)
    return MaskedSamplingResult(clean, masks, condition.lead_graph.counts,
                                nfe, tuple(states), times if return_trace else None)


def decode_masked_edit_result(model: MaskedEditModel, result: SamplingResult,
                              condition: MaskedEditCondition) -> GraphLogits:
    """One terminal decoder call; B/C force protected atom and bond categories."""
    return decode_masked_edit(model, result.latents, condition)
