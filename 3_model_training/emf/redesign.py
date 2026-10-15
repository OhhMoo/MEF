"""Evidence-conditioned molecular redesign using the existing ELF graph trunk.

The factorized model predicts F + B a. Chemical F/B see noisy state, time,
lead chemistry and candidate count; only a receives observations and goals.
The ordinary conditional variant injects those task inputs into the trunk.
Both variants share the goal-free evidence encoder and joint response model.
Attributed ELF layers are imported from elf_layers.py without duplicating them.
"""

import copy
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .contracts import GraphBatch, GraphMasks, Latents
from .elf_layers import FinalLayer, _make_linear
from .models import EMFModel, GraphFlow
from .redesign_contracts import FlowCondition, RedesignConfig, SupportBatch
from .response import EvidenceEncoder, ResponseModel


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask[..., None], values.float(), 0).sum(1) / mask.sum(1).clamp_min(1)[:, None]


def pool_latents(latents: Latents, masks: GraphMasks, max_nodes: int) -> Tensor:
    """Invariant node/pair means plus heavy-atom count divided by storage cap."""
    return torch.cat([_masked_mean(latents.nodes, masks.nodes),
                      _masked_mean(latents.pairs, masks.pairs),
                      masks.nodes.sum(1).float()[:, None] / max_nodes], -1)


def _task(condition: FlowCondition) -> Tensor:
    return torch.cat([condition.baseline.float(), condition.context.float(),
                      condition.response_state.float(), condition.goal.float()], -1)


class FlowParts(NamedTuple):
    base: Latents
    node_directions: Tensor  # [batch,nodes,node_dim,rank]
    pair_directions: Tensor  # [batch,pairs,pair_dim,rank]
    strengths: Tensor       # [batch,rank]

    def compose(self) -> Latents:
        with torch.autocast(device_type=self.base.nodes.device.type, enabled=False):
            return Latents(self.base.nodes.float() + torch.einsum('bndr,br->bnd', self.node_directions.float(), self.strengths.float()),
                           self.base.pairs.float() + torch.einsum('bpdr,br->bpd', self.pair_directions.float(), self.strengths.float()))


class _LeadFlow(GraphFlow):
    """Retain the existing projections, backbone, and clean-prediction heads."""

    def __init__(self, config: RedesignConfig, base: GraphFlow):
        super().__init__(config.model)
        self.load_state_dict(base.state_dict())
        self.graph_config = config.model
        self.lead_projection = _make_linear(config.feature_dim + 1, config.model.hidden_dim,
                                            kernel_init=nn.init.zeros_)

    def hidden(self, latents: Latents, t: Tensor, masks: GraphMasks, lead: Tensor,
               task_context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=latents.nodes.device.type, enabled=False):
            count = masks.nodes.sum(1).float()[:, None] / self.graph_config.max_nodes
            context = self.time_embedding(t.float()) + self.lead_projection(torch.cat([lead.float(), count], -1))
            if task_context is not None:
                context = context + task_context.float()
            nodes = self.node_input(latents.nodes.float()) + context[:, None]
            pairs = self.pair_input(latents.pairs.float()) + context[:, None]
        return self.backbone(nodes, pairs, masks)

    def destination(self, nodes: Tensor, pairs: Tensor, masks: GraphMasks) -> Latents:
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            return Latents(self.node_head(nodes.float()), self.pair_head(pairs.float())).masked(masks)


class FactorizedFlow(_LeadFlow):
    """One shared chemical trunk supplies F and r equivariant directions."""

    def __init__(self, config: RedesignConfig, base: GraphFlow):
        super().__init__(config, base)
        graph, self.rank = config.model, config.rank
        self.node_basis = FinalLayer(graph.hidden_dim, graph.node_dim * config.rank)
        self.pair_basis = FinalLayer(graph.hidden_dim, graph.pair_dim * config.rank)
        # ELF FinalLayer starts at zero: override B so a can learn immediately.
        nn.init.normal_(self.node_basis.linear.weight, std=.02)
        nn.init.normal_(self.pair_basis.linear.weight, std=.02)
        selector_input = 2 * graph.hidden_dim + config.feature_dim + config.context_dim + config.state_dim + 6
        self.selector = nn.Sequential(_make_linear(selector_input, graph.hidden_dim), nn.SiLU(),
                                      _make_linear(graph.hidden_dim, graph.hidden_dim), nn.SiLU(),
                                      _make_linear(graph.hidden_dim, config.rank, kernel_init=nn.init.zeros_))

    def decompose(self, latents: Latents, t: Tensor, masks: GraphMasks, condition: FlowCondition) -> FlowParts:
        nodes, pairs = self.hidden(latents, t, masks, condition.lead_features)
        base = self.destination(nodes, pairs, masks)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            node_basis = self.node_basis(nodes.float()).reshape(*nodes.shape[:2], -1, self.rank)
            pair_basis = self.pair_basis(pairs.float()).reshape(*pairs.shape[:2], self.graph_config.pair_dim, self.rank)
            node_basis = node_basis * masks.nodes[..., None, None]
            pair_basis = pair_basis * masks.pairs[..., None, None]
            selector_inputs = torch.cat([_masked_mean(nodes, masks.nodes), _masked_mean(pairs, masks.pairs),
                                         condition.lead_features.float(), _task(condition),
                                         t.float()[:, None],
                                         masks.nodes.sum(1).float()[:, None] / self.graph_config.max_nodes], -1)
            strengths = self.selector(selector_inputs)
        return FlowParts(base, node_basis, pair_basis, strengths)

    def forward(self, latents: Latents, t: Tensor, masks: GraphMasks, condition: FlowCondition) -> Latents:
        return self.decompose(latents, t, masks, condition).compose().masked(masks)


class ConditionalFlow(_LeadFlow):
    """Ordinary conditional clean prediction, with a comparable parameter count."""

    def __init__(self, config: RedesignConfig, base: GraphFlow):
        super().__init__(config, base)
        graph, rank = config.model, config.rank
        selector_input = 2 * graph.hidden_dim + config.feature_dim + config.context_dim + config.state_dim + 6
        target = ((graph.hidden_dim + 1) * rank * (graph.node_dim + graph.pair_dim) + 2 * graph.hidden_dim
                  + (selector_input + 1) * graph.hidden_dim + (graph.hidden_dim + 1) * graph.hidden_dim
                  + (graph.hidden_dim + 1) * rank)
        task_input = config.context_dim + config.state_dim + 4
        linear_term = task_input + graph.hidden_dim + 2
        width = max(1, round((math.sqrt(linear_term**2 + 4 * (target - graph.hidden_dim)) - linear_term) / 2))
        self.task_projection = nn.Sequential(_make_linear(task_input, width), nn.SiLU(),
                                             _make_linear(width, width), nn.SiLU(),
                                             _make_linear(width, graph.hidden_dim, kernel_init=nn.init.zeros_))

    def forward(self, latents: Latents, t: Tensor, masks: GraphMasks, condition: FlowCondition) -> Latents:
        with torch.autocast(device_type=latents.nodes.device.type, enabled=False):
            task_context = self.task_projection(_task(condition))
        nodes, pairs = self.hidden(latents, t, masks, condition.lead_features, task_context)
        return self.destination(nodes, pairs, masks)


class RedesignModel(nn.Module):
    """Source-trained functions; target calibration changes evidence alone."""

    def __init__(self, config: RedesignConfig, representation: EMFModel | None = None):
        super().__init__()
        self.redesign_config, self.config = config, config.model
        if representation is None:
            representation = EMFModel(config.model)
        if representation.config != config.model:
            raise ValueError("representation and redesign graph architecture differ")
        # Independent copies keep paired source runs from sharing trainable q.
        self.encoder = copy.deepcopy(representation.encoder)
        self.normalizer = copy.deepcopy(representation.normalizer)
        self.decoder = copy.deepcopy(representation.decoder).requires_grad_(True)
        self.evidence = EvidenceEncoder(config.feature_dim, config.context_dim, config.state_dim,
                                        config.model.hidden_dim)
        self.response = ResponseModel(config.feature_dim, config.context_dim, config.state_dim,
                                      config.model.hidden_dim, config.covariance_floor)
        flow_type = FactorizedFlow if config.variant == 'factorized' else ConditionalFlow
        self.flow = flow_type(config, representation.flow)
        self._deployment_frozen = False
        self.freeze_representation()

    def encode(self, batch: GraphBatch) -> Latents:
        # E is one frozen numerical map for caches, source targets and deployment.
        with torch.autocast(device_type=batch.atoms.device.type, enabled=False):
            return self.normalizer(self.encoder(batch), batch.masks)

    def decode(self, latents: Latents, masks: GraphMasks):
        return self.decoder(self.normalizer.inverse(latents, masks), masks)

    @torch.no_grad()
    def graph_features(self, batch: GraphBatch) -> Tensor:
        return pool_latents(self.encode(batch), batch.masks, self.config.max_nodes)

    def support_state(self, support: SupportBatch) -> Tensor:
        return self.evidence(support.features, support.values, support.context, support.mask)

    def condition(self, lead: GraphBatch, baseline: Tensor, context: Tensor,
                  state: Tensor, goal: Tensor) -> FlowCondition:
        return FlowCondition(self.graph_features(lead), baseline, context, state, goal)

    def predict_clean(self, z: Latents, t: Tensor, masks: GraphMasks,
                      condition: FlowCondition) -> Latents:
        return self.flow(z, t, masks, condition)

    def predict_response_features(self, lead_features: Tensor, candidate_features: Tensor, baseline: Tensor,
                                  context: Tensor, state: Tensor):
        return self.response(lead_features, baseline, candidate_features, context, state)

    def predict_response(self, lead: GraphBatch, candidate: GraphBatch, baseline: Tensor,
                         context: Tensor, state: Tensor):
        return self.predict_response_features(self.graph_features(lead), self.graph_features(candidate),
                                               baseline, context, state)

    def freeze_representation(self) -> None:
        self.encoder.requires_grad_(False).eval()
        self.normalizer.requires_grad_(False).eval()
        for parameter in self.encoder.parameters():
            parameter.grad = None

    def freeze_deployment(self) -> None:
        self._deployment_frozen = True
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self.eval()

    def train(self, mode: bool = True):
        super().train(False if self._deployment_frozen else mode)
        self.encoder.eval()
        self.normalizer.eval()
        return self
