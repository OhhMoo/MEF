"""Full-lead, fixed-alignment flow variants for the masked-edit study.

The graph backbone and terminal decoder come from the ELF-derived EMF model.
Editable masks are distinct from graph padding masks. A is an unclamped
lead-relative control; B uses absolute destinations; C uses lead-relative
destinations. B and C are hard-clamped by the sampler and terminal decode.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import GraphBatch, GraphLogits, GraphMasks, Latents
from .elf_layers import FinalLayer, _make_linear
from .models import EMFModel, GraphFlow
from .redesign import RedesignModel, pool_latents
from .redesign_contracts import RedesignConfig


def validate_edit_masks(edit: GraphMasks, valid: GraphMasks) -> None:
    if (edit.nodes.shape != valid.nodes.shape or edit.pairs.shape != valid.pairs.shape
            or edit.nodes.dtype != torch.bool or edit.pairs.dtype != torch.bool):
        raise ValueError("editable masks must be boolean and match graph mask shapes")
    if (torch.any(edit.nodes & ~valid.nodes) or torch.any(edit.pairs & ~valid.pairs)):
        raise ValueError("editable positions must be active graph positions")


def hybrid_latents(lead: Latents, query: Latents, edit: GraphMasks) -> Latents:
    """Use lead encoder coordinates at protected positions, even if E is contextual."""
    return Latents(torch.where(edit.nodes[..., None], query.nodes, lead.nodes),
                   torch.where(edit.pairs[..., None], query.pairs, lead.pairs))


@dataclass(frozen=True)
class MaskedEditCondition:
    lead_features: Tensor
    baseline: Tensor
    context: Tensor
    response_state: Tensor
    goal: Tensor
    lead: Latents
    lead_graph: GraphBatch
    edit_masks: GraphMasks

    @property
    def masks(self) -> GraphMasks:
        return self.lead_graph.masks

    def validate(self, masks: GraphMasks) -> None:
        if (not torch.equal(self.masks.nodes, masks.nodes)
                or not torch.equal(self.masks.pairs, masks.pairs)):
            raise ValueError("lead and query must have identical aligned active masks")
        validate_edit_masks(self.edit_masks, masks)

    def to(self, device) -> "MaskedEditCondition":
        return MaskedEditCondition(
            self.lead_features.to(device), self.baseline.to(device), self.context.to(device),
            self.response_state.to(device), self.goal.to(device), self.lead.to(device),
            self.lead_graph.to(device), self.edit_masks.to(device))

    def index_select(self, index: Tensor) -> "MaskedEditCondition":
        graph = self.lead_graph
        selected = GraphBatch(graph.atoms.index_select(0, index), graph.bonds.index_select(0, index),
                              GraphMasks(graph.masks.nodes.index_select(0, index),
                                         graph.masks.pairs.index_select(0, index)))
        return MaskedEditCondition(
            *(getattr(self, name).index_select(0, index) for name in
              ("lead_features", "baseline", "context", "response_state", "goal")),
            Latents(self.lead.nodes.index_select(0, index), self.lead.pairs.index_select(0, index)),
            selected, GraphMasks(self.edit_masks.nodes.index_select(0, index),
                                 self.edit_masks.pairs.index_select(0, index)))

    def expand(self, size: int) -> "MaskedEditCondition":
        if self.baseline.shape[0] != 1:
            raise ValueError("expand requires one lead condition")
        graph = self.lead_graph
        return MaskedEditCondition(
            *(getattr(self, name).expand(size, -1) for name in
              ("lead_features", "baseline", "context", "response_state", "goal")),
            Latents(self.lead.nodes.expand(size, -1, -1), self.lead.pairs.expand(size, -1, -1)),
            GraphBatch(graph.atoms.expand(size, -1), graph.bonds.expand(size, -1),
                       GraphMasks(graph.masks.nodes.expand(size, -1),
                                  graph.masks.pairs.expand(size, -1))),
            GraphMasks(self.edit_masks.nodes.expand(size, -1),
                       self.edit_masks.pairs.expand(size, -1)))


class MaskedEditFlow(GraphFlow):
    """Same full lead and edit visibility in all three flow parameterizations."""

    def __init__(self, config: RedesignConfig, base: GraphFlow, variant: str,
                 *, task_conditioned: bool = True):
        if variant not in {"A", "B", "C"}:
            raise ValueError("variant must be A, B, or C")
        super().__init__(config.model)
        self.load_state_dict(base.state_dict())
        self.variant = variant
        self.task_conditioned = task_conditioned
        graph = config.model
        self.node_lead_input = _make_linear(graph.node_dim + 1, graph.hidden_dim)
        self.pair_lead_input = _make_linear(graph.pair_dim + 1, graph.hidden_dim)
        task_dim = config.context_dim + config.state_dim + 4
        self.task_projection = nn.Sequential(
            _make_linear(task_dim, graph.hidden_dim), nn.SiLU(),
            _make_linear(graph.hidden_dim, graph.hidden_dim))
        if variant != "B":
            self.node_head = FinalLayer(graph.hidden_dim, graph.node_dim)
            self.pair_head = FinalLayer(graph.hidden_dim, graph.pair_dim)

    def forward(self, latents: Latents, t: Tensor, masks: GraphMasks,
                condition: MaskedEditCondition) -> Latents:
        lead, edit = condition.lead, condition.edit_masks
        task = torch.cat((condition.baseline.float(), condition.context.float(),
                          condition.response_state.float(), condition.goal.float()), -1)
        if not self.task_conditioned:
            task = torch.zeros_like(task)
        state = latents if self.variant == "B" else Latents(
            latents.nodes - lead.nodes, latents.pairs - lead.pairs)
        with torch.autocast(device_type=latents.nodes.device.type, enabled=False):
            common = self.time_embedding(t.float()) + self.task_projection(task)
            nodes = (self.node_input(state.nodes.float())
                     + self.node_lead_input(torch.cat((lead.nodes.float(),
                                                       edit.nodes[..., None].float()), -1))
                     + common[:, None])
            pairs = (self.pair_input(state.pairs.float())
                     + self.pair_lead_input(torch.cat((lead.pairs.float(),
                                                       edit.pairs[..., None].float()), -1))
                     + common[:, None])
        nodes, pairs = self.backbone(nodes, pairs, masks)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            out = Latents(self.node_head(nodes.float()), self.pair_head(pairs.float()))
            if self.variant != "B":
                out = Latents(lead.nodes.float() + out.nodes, lead.pairs.float() + out.pairs)
            return out.masked(masks)


class MaskedEditModel(RedesignModel):
    def __init__(self, config: RedesignConfig, representation: EMFModel,
                 *, variant: str = "C", task_conditioned: bool = True):
        super().__init__(config, representation)
        self.variant = variant
        self.flow = MaskedEditFlow(config, representation.flow, variant,
                                   task_conditioned=task_conditioned)

    def condition(self, lead: GraphBatch, baseline: Tensor, context: Tensor,
                  response_state: Tensor, goal: Tensor,
                  edit_masks: GraphMasks) -> MaskedEditCondition:
        validate_edit_masks(edit_masks, lead.masks)
        with torch.no_grad():
            encoded = self.encode(lead)
            features = pool_latents(encoded, lead.masks, self.config.max_nodes)
        condition = MaskedEditCondition(features, baseline, context, response_state,
                                        goal, encoded, lead, edit_masks)
        condition.validate(lead.masks)
        return condition


def clamp_graph_logits(logits: GraphLogits, condition: MaskedEditCondition) -> GraphLogits:
    """Protected lead categories win argmax, including no-bond categories."""
    atom_fixed = torch.full_like(logits.atoms, -1e9).scatter_(-1, condition.lead_graph.atoms[..., None], 0.)
    bond_fixed = torch.full_like(logits.bonds, -1e9).scatter_(-1, condition.lead_graph.bonds[..., None], 0.)
    return GraphLogits(torch.where(condition.edit_masks.nodes[..., None], logits.atoms, atom_fixed),
                       torch.where(condition.edit_masks.pairs[..., None], logits.bonds, bond_fixed))


def decode_masked_edit(model: MaskedEditModel, clean: Latents,
                       condition: MaskedEditCondition) -> GraphLogits:
    logits = model.decode(clean, condition.masks)
    return logits if model.variant == "A" else clamp_graph_logits(logits, condition)
