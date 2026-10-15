"""Paired fixed-noise flow probes for local-edit and factorized generators."""

from dataclasses import replace

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from .contracts import GraphMasks, Latents


def _rms_difference(a: Latents, b: Latents, masks: GraphMasks) -> list[float]:
    node_error = (a.nodes.float() - b.nodes.float()).square() * masks.nodes[..., None]
    pair_error = (a.pairs.float() - b.pairs.float()).square() * masks.pairs[..., None]
    dimensions = (masks.nodes.sum(1) * a.nodes.shape[-1]
                  + masks.pairs.sum(1) * a.pairs.shape[-1]).clamp_min(1)
    return ((node_error.sum((1, 2)) + pair_error.sum((1, 2))) / dimensions).sqrt().tolist()


def _logit_comparison(left, right, masks: GraphMasks) -> dict:
    node_dims = (masks.nodes.sum(1) * left.atoms.shape[-1]).clamp_min(1)
    pair_dims = (masks.pairs.sum(1) * left.bonds.shape[-1]).clamp_min(1)
    atom_rms = ((((left.atoms.float() - right.atoms.float()).square()
                  * masks.nodes[..., None]).sum((1, 2))) / node_dims).sqrt()
    bond_rms = ((((left.bonds.float() - right.bonds.float()).square()
                  * masks.pairs[..., None]).sum((1, 2))) / pair_dims).sqrt()
    atom_flips = ((left.atoms.argmax(-1) != right.atoms.argmax(-1)) & masks.nodes).sum(1)
    bond_flips = ((left.bonds.argmax(-1) != right.bonds.argmax(-1)) & masks.pairs).sum(1)
    return {"atom_logit_rms_difference": atom_rms.tolist(),
            "bond_logit_rms_difference": bond_rms.tolist(),
            "atom_category_flips": atom_flips.tolist(),
            "bond_category_flips": bond_flips.tolist()}


def _condition_check(positive, negative, batch: int) -> None:
    for name in ("lead_features", "baseline", "context", "goal"):
        left, right = getattr(positive, name), getattr(negative, name)
        if left.shape[0] != batch or right.shape != left.shape or not torch.equal(left, right):
            raise ValueError(f"paired conditions must share {name}")
    if positive.response_state.shape != negative.response_state.shape:
        raise ValueError("paired response states require equal shapes")
    if hasattr(positive, "lead") or hasattr(negative, "lead"):
        if not (hasattr(positive, "lead") and hasattr(negative, "lead")):
            raise ValueError("paired conditions must use the same condition type")
        for name in ("nodes", "pairs"):
            if not torch.equal(getattr(positive.lead, name), getattr(negative.lead, name)):
                raise ValueError("paired conditions must share full lead latents")
            if not torch.equal(getattr(positive.lead_masks, name),
                               getattr(negative.lead_masks, name)):
                raise ValueError("paired conditions must share lead masks")


def _selector_diagnostics(model, state: Latents, t: Tensor, masks: GraphMasks,
                          positive, negative) -> dict:
    flow = model.flow
    if not hasattr(flow, "decompose"):
        return {}
    selector = getattr(flow.selector, "selector", flow.selector)
    activations = []
    hook = None
    if hasattr(selector, "__getitem__") and len(selector) > 3:
        def capture(_module, args):
            values = args[0].detach().float()
            activations.append({
                "second_preactivation_min": float(values.min()),
                "second_preactivation_max": float(values.max()),
                "second_preactivation_below_minus_20_fraction": float((values < -20).float().mean()),
            })
        hook = selector[3].register_forward_pre_hook(capture)
    try:
        left = flow.decompose(state, t, masks, positive)
        right = flow.decompose(state, t, masks, negative)
    finally:
        if hook is not None:
            hook.remove()
    # Form B a directly.  Computing (F+B a)-F in float32 can erase the
    # extremely small branch measured in older factorized checkpoints.
    left_correction = Latents(
        torch.einsum("bndr,br->bnd", left.node_directions.float(), left.strengths.float()),
        torch.einsum("bpdr,br->bpd", left.pair_directions.float(), left.strengths.float()))
    right_correction = Latents(
        torch.einsum("bndr,br->bnd", right.node_directions.float(), right.strengths.float()),
        torch.einsum("bpdr,br->bpd", right.pair_directions.float(), right.strengths.float()))
    left_basis = Latents(left.node_directions.flatten(-2), left.pair_directions.flatten(-2))
    right_basis = Latents(right.node_directions.flatten(-2), right.pair_directions.flatten(-2))
    return {"base_rms_difference": _rms_difference(left.base, right.base, masks),
            "basis_rms_difference": _rms_difference(left_basis, right_basis, masks),
            "composed_correction_rms_difference": _rms_difference(left_correction, right_correction, masks),
            "selector_strength_rms_difference":
                (left.strengths.float() - right.strengths.float()).square().mean(-1).sqrt().tolist(),
            "selector_activation": activations}


def _state_jvp(model, state: Latents, t: Tensor, masks: GraphMasks, condition) -> list[float]:
    if condition.response_state.shape[1] < 1:
        raise ValueError("response_state needs a first coordinate for JVP")
    direction = torch.zeros_like(condition.response_state)
    direction[:, 0] = 1

    def prediction(response_state):
        updated = replace(condition, response_state=response_state)
        clean = model.predict_clean(state, t, masks, updated)
        return torch.cat((clean.nodes.flatten(1), clean.pairs.flatten(1)), -1)

    # Fused SDPA lacks forward AD on some CPU/CUDA builds.  The math kernel
    # computes the same attention function and supports the diagnostic JVP.
    # Expanded single-lead conditions have zero-stride response states.
    # Forward AD requires independent storage for every element.
    primal = condition.response_state.detach().clone().contiguous()
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        _, derivative = torch.func.jvp(
            prediction, (primal,), (direction,))
    node_size = state.nodes[0].numel()
    derivative = Latents(derivative[:, :node_size].reshape_as(state.nodes),
                         derivative[:, node_size:].reshape_as(state.pairs))
    zero = Latents(torch.zeros_like(state.nodes), torch.zeros_like(state.pairs))
    return _rms_difference(derivative, zero, masks)


def _canonical_outcomes(model, clean: Latents, masks: GraphMasks, counts: Tensor) -> list[dict]:
    from .chem import decode_graph

    logits = model.decode(clean, masks)
    atoms = logits.atoms.argmax(-1).cpu().numpy()
    bonds = logits.bonds.argmax(-1).cpu().numpy()
    return [{"smiles": outcome.smiles, "valid": outcome.chemically_valid,
             "connected": outcome.connected, "failure_reason": outcome.failure_reason}
            for outcome in (decode_graph(a, b, int(count), model.config.max_nodes)
                            for a, b, count in zip(atoms, bonds, counts.cpu().tolist()))]


def trace_pair(model, counts: Tensor, positive_condition, negative_condition,
               initial_noise: Latents, *, nfe: int = 32, T: float = .95,
               include_canonical: bool = False, known_state_jvp: bool = False) -> dict:
    """Compare two contexts at shared positive states and paired full rollouts.

    ``initial_noise`` is in the model's public absolute coordinates.  For a
    local-edit model, pass ``anchored_initial_noise``.  The three probes add
    diagnostic model calls; ``nfe`` remains the budget of each full rollout.
    """
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("trace requires frozen model parameters")
    if any(module.training for module in model.modules()):
        raise ValueError("trace requires evaluation mode")
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 2 or not 0 < T < 1:
        raise ValueError("require integer nfe>=2 and 0<T<1")
    device = initial_noise.nodes.device
    counts = torch.as_tensor(counts, device=device).long()
    masks = GraphMasks.from_counts(counts, model.config.max_nodes)
    batch = len(counts)
    _condition_check(positive_condition, negative_condition, batch)
    expected_nodes = (batch, model.config.max_nodes, model.config.node_dim)
    expected_pairs = (batch, model.config.max_nodes * (model.config.max_nodes - 1) // 2,
                      model.config.pair_dim)
    if initial_noise.nodes.shape != expected_nodes or initial_noise.pairs.shape != expected_pairs:
        raise ValueError("initial_noise shape does not match model/counts")
    times = torch.linspace(0, T, nfe, device=device, dtype=torch.float32)
    probe_indices = {target: int(torch.argmin((times - target).abs()))
                     for target in (.1, .5, .9)}
    probes = {}
    endpoints = []
    with torch.no_grad():
        for path, condition in (("positive", positive_condition), ("negative", negative_condition)):
            state = initial_noise.masked(masks)
            for step in range(nfe):
                time = times[step].expand(batch)
                clean = model.predict_clean(state, time, masks, condition).masked(masks)
                if path == "positive":
                    for target, index in probe_indices.items():
                        if step != index:
                            continue
                        other = model.predict_clean(state, time, masks, negative_condition).masked(masks)
                        logits, other_logits = model.decode(clean, masks), model.decode(other, masks)
                        entry = {"time": float(times[step]),
                                 "clean_rms_difference": _rms_difference(clean, other, masks),
                                 **_logit_comparison(logits, other_logits, masks),
                                 **_selector_diagnostics(model, state, time, masks,
                                                         positive_condition, negative_condition)}
                        if hasattr(positive_condition, "lead"):
                            entry["positive_edit_rms"] = _rms_difference(
                                clean, positive_condition.lead, masks)
                            entry["negative_edit_rms"] = _rms_difference(
                                other, negative_condition.lead, masks)
                        if known_state_jvp and target == .5:
                            entry["known_state_jvp_rms"] = _state_jvp(
                                model, state, time, masks, positive_condition)
                        probes[str(target)] = entry
                if step < nfe - 1:
                    dt = times[step + 1] - times[step]
                    state = Latents(state.nodes + dt * (clean.nodes - state.nodes) / (1 - times[step]),
                                    state.pairs + dt * (clean.pairs - state.pairs) / (1 - times[step])).masked(masks)
            endpoints.append(clean)
    positive_logits = model.decode(endpoints[0], masks)
    negative_logits = model.decode(endpoints[1], masks)
    result = {"nfe_per_path": nfe, "T": T, "attempts": batch,
              "probes_same_positive_state": probes,
              "paired_endpoint": {"clean_rms_difference": _rms_difference(*endpoints, masks),
                                  **_logit_comparison(positive_logits, negative_logits, masks)}}
    if hasattr(positive_condition, "lead"):
        result["paired_endpoint"]["positive_edit_rms"] = _rms_difference(
            endpoints[0], positive_condition.lead, masks)
        result["paired_endpoint"]["negative_edit_rms"] = _rms_difference(
            endpoints[1], negative_condition.lead, masks)
    if include_canonical:
        result["paired_endpoint"]["positive_outcomes"] = _canonical_outcomes(
            model, endpoints[0], masks, counts)
        result["paired_endpoint"]["negative_outcomes"] = _canonical_outcomes(
            model, endpoints[1], masks, counts)
    return result
