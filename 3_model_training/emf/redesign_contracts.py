"""Explicit evidence, query, and deployment contracts for calibrated redesign.

Private oracle parameters, query outcomes and campaign identifiers never enter
SupportBatch or FlowCondition. Existing graph tensor contracts remain unchanged.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .contracts import GraphBatch, ModelConfig


@dataclass(frozen=True)
class RedesignConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rank: int = 8
    state_dim: int = 64
    context_dim: int = 4
    variant: str = "factorized"
    covariance_floor: float = 1e-4

    @property
    def feature_dim(self) -> int:
        return self.model.node_dim + self.model.pair_dim + 1

    def __post_init__(self):
        if min(self.rank, self.state_dim, self.context_dim) < 1:
            raise ValueError("Redesign dimensions must be positive")
        if self.variant not in {"factorized", "conditional"}:
            raise ValueError("variant must be factorized or conditional")
        if self.covariance_floor <= 0:
            raise ValueError("covariance_floor must be positive")


@dataclass(frozen=True)
class SupportBatch:
    features: Tensor     # [episodes, support_slots, feature_dim]
    values: Tensor       # [episodes, support_slots, 2]
    context: Tensor      # [episodes, context_dim], shared by each episode
    mask: Tensor         # [episodes, support_slots], True for observed records

    def to(self, device) -> "SupportBatch":
        return SupportBatch(*(getattr(self, name).to(device) for name in
                              ("features", "values", "context", "mask")))


@dataclass(frozen=True)
class FlowCondition:
    lead_features: Tensor   # [queries, feature_dim]
    baseline: Tensor        # [queries, 2]
    context: Tensor         # [queries, context_dim]
    response_state: Tensor  # [queries, state_dim]
    goal: Tensor            # [queries, 2]: improvement, preservation threshold

    def to(self, device) -> "FlowCondition":
        return FlowCondition(*(getattr(self, name).to(device) for name in
                               ("lead_features", "baseline", "context",
                                "response_state", "goal")))

    def index_select(self, index: Tensor) -> "FlowCondition":
        return FlowCondition(*(getattr(self, name).index_select(0, index) for name in
                               ("lead_features", "baseline", "context",
                                "response_state", "goal")))

    def expand(self, size: int) -> "FlowCondition":
        if self.baseline.shape[0] != 1:
            raise ValueError("expand requires a single lead condition")
        return FlowCondition(*(getattr(self, name).expand(size, -1) for name in
                               ("lead_features", "baseline", "context",
                                "response_state", "goal")))


@dataclass(frozen=True)
class CampaignBatch:
    query: GraphBatch       # [episodes * queries_per_episode]
    lead: GraphBatch        # repeated lead, one row per query
    support: SupportBatch   # one support set per episode
    episode_index: Tensor   # maps each query to its support episode
    baseline: Tensor        # [queries, 2]
    context: Tensor         # [queries, context_dim]
    goal: Tensor            # [queries, 2], sampled independently of query outcomes
    delta: Tensor           # [queries, 2], labels for the loss only
    flow_weights: Tensor    # [queries], exact measured-profile feasibility

    def to(self, device) -> "CampaignBatch":
        return CampaignBatch(
            self.query.to(device), self.lead.to(device), self.support.to(device),
            *(getattr(self, name).to(device) for name in
              ("episode_index", "baseline", "context", "goal", "delta",
               "flow_weights")),
        )
