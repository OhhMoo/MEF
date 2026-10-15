"""Lead-anchored clean prediction for graph-aligned molecular local edits.

The pinned ELF graph backbone and terminal decoder are reused.  A frozen
encoder supplies normalized, full atom/pair lead latents.  The flow predicts
an edit residual relative to those aligned latents and receives the known
campaign state, goal, and lead outcome through ordinary task conditioning.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import GraphBatch, GraphMasks, Latents
from .elf_layers import FinalLayer, _make_linear
from .models import EMFModel, GraphFlow
from .redesign import RedesignModel, pool_latents
from .redesign_contracts import FlowCondition, RedesignConfig


@dataclass(frozen=True)
class LocalEditCondition(FlowCondition):
    """Full lead representation in the data builder's lead atom order.

    The first study uses fixed-size, atom-aligned graphs.  Alignment is an
    explicit data operation before constructing this condition; a candidate
    with a different atom order or count must not be passed here.
    """

    lead: Latents             # normalized full lead in candidate atom order
    lead_masks: GraphMasks

    def to(self, device) -> "LocalEditCondition":
        return LocalEditCondition(
            *(getattr(self, field).to(device) for field in
              ("lead_features", "baseline", "context", "response_state", "goal")),
            self.lead.to(device), self.lead_masks.to(device))

    def index_select(self, index: Tensor) -> "LocalEditCondition":
        return LocalEditCondition(
            *(getattr(self, field).index_select(0, index) for field in
              ("lead_features", "baseline", "context", "response_state", "goal")),
            Latents(self.lead.nodes.index_select(0, index), self.lead.pairs.index_select(0, index)),
            GraphMasks(self.lead_masks.nodes.index_select(0, index),
                       self.lead_masks.pairs.index_select(0, index)))

    def expand(self, size: int) -> "LocalEditCondition":
        if self.baseline.shape[0] != 1:
            raise ValueError("expand requires one lead condition")
        return LocalEditCondition(
            *(getattr(self, field).expand(size, -1) for field in
              ("lead_features", "baseline", "context", "response_state", "goal")),
            Latents(self.lead.nodes.expand(size, -1, -1),
                    self.lead.pairs.expand(size, -1, -1)),
            GraphMasks(self.lead_masks.nodes.expand(size, -1),
                       self.lead_masks.pairs.expand(size, -1)))

    def aligned_lead(self, masks: GraphMasks, *, validate: bool = True) -> Latents:
        if (self.lead.nodes.shape[:2] != masks.nodes.shape
                or self.lead.pairs.shape[:2] != masks.pairs.shape):
            raise ValueError("lead and candidate require the same aligned atom count and pair order")
        if validate and (not torch.equal(self.lead_masks.nodes, masks.nodes)
                         or not torch.equal(self.lead_masks.pairs, masks.pairs)):
            raise ValueError("lead and candidate require identical active masks")
        return self.lead.masked(masks)


class LocalEditFlow(GraphFlow):
    """ELF graph trunk with full lead tokens and an additive edit destination."""

    def __init__(self, config: RedesignConfig, base: GraphFlow, *, task_conditioned: bool = True):
        super().__init__(config.model)
        self.load_state_dict(base.state_dict())
        graph = config.model
        self.task_conditioned = task_conditioned
        self.node_lead_input = _make_linear(graph.node_dim + 1, graph.hidden_dim)
        self.pair_lead_input = _make_linear(graph.pair_dim + 1, graph.hidden_dim)
        task_dim = config.context_dim + config.state_dim + 4
        self.task_projection = nn.Sequential(
            _make_linear(task_dim, graph.hidden_dim), nn.SiLU(),
            _make_linear(graph.hidden_dim, graph.hidden_dim))
        # Existing flow heads are replaced by zero-start residual heads.  At
        # initialization the destination is the mapped lead itself.
        self.node_head = FinalLayer(graph.hidden_dim, graph.node_dim)
        self.pair_head = FinalLayer(graph.hidden_dim, graph.pair_dim)

    def forward(self, latents: Latents, t: Tensor, masks: GraphMasks,
                condition: LocalEditCondition) -> Latents:
        # Mask equality is checked once when the anchored initial state or
        # training objective is built.  Avoid a device synchronization here
        # on every flow evaluation.
        aligned = condition.aligned_lead(masks, validate=False)
        residual = Latents(latents.nodes - aligned.nodes,
                           latents.pairs - aligned.pairs).masked(masks)
        task = torch.cat((condition.baseline.float(), condition.context.float(),
                          condition.response_state.float(), condition.goal.float()), -1)
        if not self.task_conditioned:
            task = torch.zeros_like(task)
        with torch.autocast(device_type=latents.nodes.device.type, enabled=False):
            common = self.time_embedding(t.float()) + self.task_projection(task)
            nodes = (self.node_input(residual.nodes.float())
                     + self.node_lead_input(torch.cat((aligned.nodes.float(),
                                                      masks.nodes[..., None].float()), -1))
                     + common[:, None])
            pairs = (self.pair_input(residual.pairs.float())
                     + self.pair_lead_input(torch.cat((aligned.pairs.float(),
                                                      masks.pairs[..., None].float()), -1))
                     + common[:, None])
        nodes, pairs = self.backbone(nodes, pairs, masks)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            return Latents(aligned.nodes.float() + self.node_head(nodes.float()),
                           aligned.pairs.float() + self.pair_head(pairs.float())).masked(masks)


class LocalEditModel(RedesignModel):
    """Known-context local generator with frozen continuous encoder."""

    def __init__(self, config: RedesignConfig, representation: EMFModel,
                 *, task_conditioned: bool = True):
        super().__init__(config, representation)
        self.flow = LocalEditFlow(config, representation.flow, task_conditioned=task_conditioned)

    def condition(self, lead: GraphBatch, baseline: Tensor, context: Tensor,
                  response_state: Tensor, goal: Tensor) -> LocalEditCondition:
        encoded = self.encode(lead)
        return LocalEditCondition(
            pool_latents(encoded, lead.masks, self.config.max_nodes),
            baseline, context, response_state, goal, encoded, lead.masks)

