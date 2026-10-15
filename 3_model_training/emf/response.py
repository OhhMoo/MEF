"""Goal-free evidence inference and a normalized joint experimental response."""

import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F

from .elf_layers import _make_linear


class EvidenceEncoder(nn.Module):
    """DeepSets over measured records; record order has no meaning.

    The set size is supplied alongside its mean representation so additional
    measurements can change the inferred state. No campaign-ID or goal input
    exists. Frozen molecular features are supplied by the redesign model.
    """

    def __init__(self, feature_dim: int, context_dim: int, state_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.record = nn.Sequential(_make_linear(feature_dim + 2 + context_dim, hidden_dim),
                                    nn.SiLU(), _make_linear(hidden_dim, hidden_dim), nn.SiLU())
        self.summary = nn.Sequential(_make_linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
                                     _make_linear(hidden_dim, state_dim))

    def forward(self, features: Tensor, values: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 3 or values.shape != (*features.shape[:2], 2) or mask.shape != features.shape[:2]:
            raise ValueError("evidence requires [episodes,records,features], two outcomes, and a record mask")
        with torch.autocast(device_type=features.device.type, enabled=False):
            mask = mask.bool()
            repeated_context = context.float()[:, None].expand(-1, features.shape[1], -1)
            inputs = torch.cat([features.float(), values.float(), repeated_context], -1)
            # Invalid storage entries cannot enter the MLP or contaminate gradients.
            inputs = torch.where(mask[..., None], inputs, 0)
            encoded = self.record(inputs)
            count = mask.sum(1).float()
            mean = torch.where(mask[..., None], encoded, 0).sum(1) / count.clamp_min(1)[:, None]
            return self.summary(torch.cat([mean, count.log1p()[:, None]], -1))


class ResponseModel(nn.Module):
    """Predict p(delta | lead, baseline, candidate, context, evidence).

    The learned lower-triangular matrix L gives covariance LL^T + 1e-4 I.
    This is one normalized two-dimensional Gaussian, including endpoint
    dependence and observation uncertainty, in source-normalized outcome units.
    """

    def __init__(self, feature_dim: int, context_dim: int, state_dim: int = 64,
                 hidden_dim: int = 128, covariance_floor: float = 1e-4):
        super().__init__()
        if covariance_floor <= 0:
            raise ValueError("covariance floor must be positive")
        self.covariance_floor = float(covariance_floor)
        self.network = nn.Sequential(
            _make_linear(3 * feature_dim + 2 + context_dim + state_dim, hidden_dim), nn.SiLU(),
            _make_linear(hidden_dim, hidden_dim), nn.SiLU(), _make_linear(hidden_dim, 5))

    def forward(self, lead: Tensor, baseline: Tensor, candidate: Tensor,
                context: Tensor, state: Tensor) -> MultivariateNormal:
        with torch.autocast(device_type=lead.device.type, enabled=False):
            lead, candidate = lead.float(), candidate.float()
            raw = self.network(torch.cat([lead, candidate, candidate - lead,
                                          baseline.float(), context.float(), state.float()], -1))
            mean = raw[..., :2]
            diagonal = F.softplus(raw[..., 2:4]) + 1e-4
            zero = torch.zeros_like(diagonal[..., 0])
            lower = torch.stack([diagonal[..., 0], zero, raw[..., 4], diagonal[..., 1]], -1)
            lower = lower.reshape(*mean.shape[:-1], 2, 2)
            covariance = lower @ lower.transpose(-1, -2)
            covariance = covariance + self.covariance_floor * torch.eye(2, device=lead.device, dtype=torch.float32)
            return MultivariateNormal(mean, covariance_matrix=covariance)
