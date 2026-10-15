"""Source-trained soft-graph descriptor critic and known-coefficient control.

The critic approximates source-scaled RDKit MolLogP and TPSA from terminal
decoder probabilities. Control uses only that critic and supplied coefficients;
its two response inequalities are linearized locally and audited by rerollout.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import GraphBatch, GraphLogits, GraphMasks, Latents, ModelConfig
from .local_edit_control import _minimum_norm_two_constraints
from .masked_edit_model import MaskedEditCondition, MaskedEditModel
from .masked_edit_sampling import (MaskedSamplingResult, decode_masked_edit_result,
                                   sample_masked_edit_latents)
from .models import GraphBackbone


def hard_graph_probabilities(graph: GraphBatch) -> GraphLogits:
    """Represent observed source graphs as differentiable one-hot inputs."""
    return GraphLogits(F.one_hot(graph.atoms.long(), 4).float(),
                       F.one_hot(graph.bonds.long(), 4).float())


class SoftGraphDescriptorCritic(nn.Module):
    """Two-layer graph-aware regression from category probabilities to u,q."""

    def __init__(self, config: ModelConfig, *, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        if hidden_dim < 4 or depth < 1 or hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four and depth positive")
        self.config = config
        self.hidden_dim = hidden_dim
        self.depth = depth
        graph_config = ModelConfig(max_nodes=config.max_nodes, node_dim=hidden_dim,
                                   pair_dim=hidden_dim, hidden_dim=hidden_dim,
                                   depth=depth, num_heads=4)
        self.node_embedding = nn.Parameter(torch.randn(4, hidden_dim) * .02)
        self.pair_embedding = nn.Parameter(torch.randn(4, hidden_dim) * .02)
        self.backbone = GraphBackbone(graph_config)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden_dim + 1),
                                  nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, 2))

    def forward(self, graph: GraphLogits, masks: GraphMasks,
                *, probabilities: bool = False) -> Tensor:
        if (graph.atoms.shape[:2] != masks.nodes.shape
                or graph.bonds.shape[:2] != masks.pairs.shape
                or graph.atoms.shape[-1] != 4 or graph.bonds.shape[-1] != 4):
            raise ValueError("graph categories and masks have incompatible shapes")
        if probabilities:
            atoms, bonds = graph.atoms.float(), graph.bonds.float()
            if (not torch.isfinite(atoms).all() or not torch.isfinite(bonds).all()
                    or (atoms < 0).any() or (bonds < 0).any()):
                raise ValueError("probabilities must be finite and nonnegative")
            # Source one-hot and generated soft probabilities use the same path.
            atoms = atoms / atoms.sum(-1, keepdim=True).clamp_min(1e-8)
            bonds = bonds / bonds.sum(-1, keepdim=True).clamp_min(1e-8)
        else:
            atoms, bonds = graph.atoms.float().softmax(-1), graph.bonds.float().softmax(-1)
        nodes = atoms @ self.node_embedding
        pairs = bonds @ self.pair_embedding
        nodes, pairs = self.backbone(nodes, pairs, masks)
        node_mean = (nodes * masks.nodes[..., None]).sum(1) / masks.nodes.sum(1).clamp_min(1)[:, None]
        pair_mean = (pairs * masks.pairs[..., None]).sum(1) / masks.pairs.sum(1).clamp_min(1)[:, None]
        count = masks.nodes.sum(1).float()[:, None] / self.config.max_nodes
        return self.head(torch.cat((node_mean, pair_mean, count), -1))


def fit_descriptor_critic(critic: SoftGraphDescriptorCritic, inputs: GraphLogits,
                          masks: GraphMasks, labels: Tensor, *, probabilities: bool = True,
                          steps: int = 500, batch_size: int = 128,
                          learning_rate: float = 1e-3, seed: int = 142) -> dict:
    """Fit source-only hard/soft graph records supplied by the caller.

    Labels are [MolLogP/scale0,TPSA/scale1]. Concatenate one-hot source graphs
    and generated soft probabilities before calling to mix both domains.
    """
    size = labels.shape[0]
    if (labels.shape != (size, 2) or inputs.atoms.shape[0] != size
            or inputs.bonds.shape[0] != size or masks.nodes.shape[0] != size
            or masks.pairs.shape[0] != size or size < 1):
        raise ValueError("inputs, masks, labels must have matching nonempty batch")
    if (not torch.isfinite(labels).all() or steps < 1 or batch_size < 1
            or learning_rate <= 0):
        raise ValueError("finite labels and positive fit budget required")
    device = next(critic.parameters()).device
    if any(x.device != device for x in
           (inputs.atoms, inputs.bonds, masks.nodes, masks.pairs, labels)):
        raise ValueError("critic and training tensors must be on the same device")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=learning_rate, weight_decay=1e-4)
    critic.train()
    losses = []
    for step in range(steps):
        index = torch.randint(size, (min(size, batch_size),), generator=rng).to(device)
        selected = GraphLogits(inputs.atoms.index_select(0, index),
                               inputs.bonds.index_select(0, index))
        selected_masks = GraphMasks(masks.nodes.index_select(0, index),
                                    masks.pairs.index_select(0, index))
        prediction = critic(selected, selected_masks, probabilities=probabilities)
        loss = (prediction - labels.index_select(0, index).float()).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == steps:
            losses.append(float(loss.detach().cpu()))
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        prediction = critic(inputs, masks, probabilities=probabilities)
        rmse = (prediction - labels).square().mean(0).sqrt()
    return {"records": size, "steps": steps, "batch_size": min(size, batch_size),
            "initial_final_batch_mse": losses, "source_rmse_scaled": rmse.cpu().tolist()}


def descriptor_features(descriptors_scaled: Tensor, lead_descriptors_scaled: Tensor) -> Tensor:
    """Match masked_edit_data._features exactly: phi(G)-phi(lead)."""
    if descriptors_scaled.shape != lead_descriptors_scaled.shape or descriptors_scaled.shape[-1] != 2:
        raise ValueError("candidate and lead scaled descriptors must be [batch,2]")
    u, q = descriptors_scaled.unbind(-1)
    lead_u, lead_q = lead_descriptors_scaled.unbind(-1)
    return torch.stack((u - lead_u, q - lead_q, u * q - lead_u * lead_q), -1)


def flip_first_response_row(coefficients: Tensor) -> Tensor:
    """Opposite-coefficient diagnostic changes only endpoint one's W row."""
    if coefficients.ndim != 3 or coefficients.shape[1:] != (2, 3):
        raise ValueError("coefficients must be [batch,2,3]")
    opposite = coefficients.clone()
    opposite[:, 0] = -opposite[:, 0]
    return opposite


@dataclass(frozen=True)
class ControlResult:
    sample: MaskedSamplingResult
    node_control: Tensor       # [batch,2,nodes,node_dim]
    pair_control: Tensor       # [batch,2,pairs,pair_dim]
    predicted_before: Tensor   # [batch,2], critic plus known W
    predicted_after: Tensor
    forward_nfe: int           # actual flow forward calls, across all rerollouts
    backward_nfe: int          # flow calls traversed by endpoint VJPs
    control_energy: Tensor     # [batch], time-integrated editable mean-square velocity
    control_rms: Tensor        # [batch], sqrt(energy / (1-start_t))
    gradient_norms: tuple[Tensor, ...]  # per iteration [batch,2]
    linearized_feasible: tuple[Tensor, ...]
    accepted_steps: Tensor     # [batch], accepted SQP iterations

    @property
    def latents(self) -> Latents:
        return self.sample.latents


@contextmanager
def _frozen_for_control(model: nn.Module, critic: nn.Module):
    parameters = [(p, p.requires_grad) for module in (model, critic) for p in module.parameters()]
    modes = [(module, module.training) for root in (model, critic) for module in root.modules()]
    model.eval()
    critic.eval()
    for parameter, _ in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in parameters:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def optimize_terminal_control(
        model: MaskedEditModel, condition: MaskedEditCondition,
        gaussian: Latents | None, critic: SoftGraphDescriptorCritic,
        coefficients: Tensor, lead_desc: Tensor, scales: Tensor, goal: Tensor,
        *, nfe: int = 16, T: float = .95, iterations: int = 4,
        radii: tuple[float, ...] = (.02, .05), ridge: float = 1e-4,
        start_t: float = 0., initial_state: Latents | None = None) -> ControlResult:
    """Sequential minimum-norm velocity repair through full terminal rollout.

    Two control bins share one endpoint Jacobian. The signed response gap is
    linearized at the current U; every clipped trial is rerolled and accepted
    only if the critic's actual two-goal violation improves. This is a local
    surrogate procedure; linearized feasibility does not guarantee chemistry
    or exact response feasibility.
    """
    batch = condition.masks.nodes.shape[0]
    if (coefficients.shape != (batch, 2, 3) or lead_desc.shape != (batch, 2)
            or scales.shape != (2,) or goal.shape != (batch, 2)):
        raise ValueError("require W[B,2,3], raw lead_desc[B,2], scales[2], goal[B,2]")
    if (not all(torch.isfinite(x).all() for x in (coefficients, lead_desc, scales, goal))
            or (scales <= 0).any()):
        raise ValueError("coefficients, descriptors, scales and goals must be finite")
    if (isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0
            or not radii or any(not math.isfinite(r) or r <= 0 for r in radii)
            or ridge <= 0):
        raise ValueError("invalid SQP iteration count, trust radii or ridge")
    if next(critic.parameters()).device != condition.lead.nodes.device:
        raise ValueError("critic and model condition must share a device")
    device = condition.lead.nodes.device
    coefficients = coefficients.to(device).float().detach()
    lead_scaled = (lead_desc.to(device).float() / scales.to(device).float()).detach()
    goal = goal.to(device).float().detach()
    node_shape = (batch, 2, *condition.lead.nodes.shape[1:])
    pair_shape = (batch, 2, *condition.lead.pairs.shape[1:])
    node_u = torch.zeros(node_shape, device=device)
    pair_u = torch.zeros(pair_shape, device=device)
    mid = (start_t + T) / 2
    forward_nfe = backward_nfe = 0
    gradient_norms: list[Tensor] = []
    feasible_history: list[Tensor] = []
    accepted_steps = torch.zeros(batch, dtype=torch.long, device=device)
    initial_prediction = None
    current_prediction = None
    current_sample = None

    def rollout(node_velocity: Tensor, pair_velocity: Tensor):
        def controller(state: Latents, time: Tensor, supplied: MaskedEditCondition):
            bin_one = (time < mid)[:, None, None]
            return Latents(torch.where(bin_one, node_velocity[:, 0], node_velocity[:, 1]),
                           torch.where(bin_one, pair_velocity[:, 0], pair_velocity[:, 1]))

        sample = sample_masked_edit_latents(
            model, condition, gaussian, nfe=nfe, T=T, start_t=start_t,
            initial_state=initial_state, velocity_controller=controller)
        logits = decode_masked_edit_result(model, sample, condition)
        descriptors = critic(logits, sample.masks)
        features = descriptor_features(descriptors, lead_scaled)
        predicted = torch.einsum("bij,bj->bi", coefficients, features)
        return sample, predicted

    with _frozen_for_control(model, critic), torch.enable_grad():
        for iteration in range(max(1, iterations)):
            node_variable = node_u.detach().requires_grad_(True)
            pair_variable = pair_u.detach().requires_grad_(True)
            sample, predicted = rollout(node_variable, pair_variable)
            forward_nfe += nfe
            if initial_prediction is None:
                initial_prediction = predicted.detach()
            current_prediction = predicted.detach()
            current_sample = sample
            if iteration >= iterations:
                break
            if bool((current_prediction >= goal).all()):
                break
            gradients = []
            for endpoint in range(2):
                gn, gp = torch.autograd.grad(predicted[:, endpoint].sum(),
                                             (node_variable, pair_variable),
                                             retain_graph=endpoint == 0,
                                             allow_unused=True)
                if gn is None:
                    gn = torch.zeros_like(node_variable)
                if gp is None:
                    gp = torch.zeros_like(pair_variable)
                gradients.append(torch.cat((gn.flatten(1), gp.flatten(1)), -1))
                backward_nfe += nfe
            jacobian = torch.stack(gradients, 1)
            gradient_norms.append(jacobian.norm(dim=-1).detach())
            deficit = goal - current_prediction
            direction, feasible = _minimum_norm_two_constraints(jacobian, deficit, ridge)
            feasible_history.append(feasible.detach())
            node_count = node_u[0].numel()
            node_step = direction[:, :node_count].reshape_as(node_u)
            pair_step = direction[:, node_count:].reshape_as(pair_u)
            node_step = node_step * condition.edit_masks.nodes[:, None, :, None]
            pair_step = pair_step * condition.edit_masks.pairs[:, None, :, None]
            coordinates = (2 * (condition.edit_masks.nodes.sum(1) * node_u.shape[-1]
                                + condition.edit_masks.pairs.sum(1) * pair_u.shape[-1])).clamp_min(1)
            step_rms = ((node_step.square().flatten(1).sum(1)
                         + pair_step.square().flatten(1).sum(1)) / coordinates).sqrt()
            best_score = (goal - current_prediction).clamp_min(0).square().sum(-1)
            best_node, best_pair = node_u, pair_u
            best_sample = current_sample
            best_prediction = current_prediction
            improved_any = torch.zeros(batch, dtype=torch.bool, device=device)
            for radius in radii:
                clip = (radius / step_rms.clamp_min(1e-20)).clamp(max=1)
                trial_node = node_u + clip[:, None, None, None] * node_step
                trial_pair = pair_u + clip[:, None, None, None] * pair_step
                with torch.no_grad():
                    trial_sample, trial_prediction = rollout(trial_node, trial_pair)
                forward_nfe += nfe
                trial_score = (goal - trial_prediction).clamp_min(0).square().sum(-1)
                improved = trial_score < best_score - 1e-10
                improved_any |= improved
                best_score = torch.where(improved, trial_score, best_score)
                best_node = torch.where(improved[:, None, None, None], trial_node, best_node)
                best_pair = torch.where(improved[:, None, None, None], trial_pair, best_pair)
                best_prediction = torch.where(improved[:, None], trial_prediction, best_prediction)
                best_sample = MaskedSamplingResult(
                    Latents(torch.where(improved[:, None, None], trial_sample.latents.nodes,
                                        best_sample.latents.nodes),
                            torch.where(improved[:, None, None], trial_sample.latents.pairs,
                                        best_sample.latents.pairs)),
                    sample.masks, sample.counts, sample.nfe)
            node_u, pair_u = best_node.detach(), best_pair.detach()
            current_prediction = best_prediction.detach()
            current_sample = best_sample
            accepted_steps += improved_any.long()
            if not bool(improved_any.any()):
                break
    assert current_sample is not None and current_prediction is not None and initial_prediction is not None
    # Match the solver quadrature exactly. Euler intervals use their left
    # evaluation times; the terminal clean correction is the remaining
    # interval [T,1] with u(T), which belongs to the second bin.
    solver_times = torch.linspace(start_t, T, nfe, device=device, dtype=torch.float32)
    bin_weights = torch.zeros(2, device=device)
    for step in range(nfe - 1):
        bin_index = 0 if bool(solver_times[step] < mid) else 1
        bin_weights[bin_index] += solver_times[step + 1] - solver_times[step]
    bin_weights[1] += 1 - T
    coordinates = (condition.edit_masks.nodes.sum(1) * node_u.shape[-1]
                   + condition.edit_masks.pairs.sum(1) * pair_u.shape[-1]).clamp_min(1)
    bin_squares = node_u.square().flatten(2).sum(-1) + pair_u.square().flatten(2).sum(-1)
    energy = (bin_squares * bin_weights[None]).sum(1) / coordinates
    control_rms = (energy / (1 - start_t)).sqrt()
    detached_sample = MaskedSamplingResult(current_sample.latents.detach(),
                                           current_sample.masks, current_sample.counts,
                                           current_sample.nfe)
    return ControlResult(detached_sample, node_u, pair_u, initial_prediction,
                         current_prediction, forward_nfe, backward_nfe,
                         energy.detach(), control_rms.detach(), tuple(gradient_norms),
                         tuple(feasible_history), accepted_steps)
