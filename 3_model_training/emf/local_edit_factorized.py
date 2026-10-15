"""Parameter-free scale repair for the older F+B a selector experiment."""

import torch
from torch import Tensor, nn

from .redesign import FactorizedFlow


class _NormalizedPooledSelector(nn.Module):
    def __init__(self, selector: nn.Module, hidden_dim: int):
        super().__init__()
        self.selector = selector
        self.hidden_dim = hidden_dim

    def forward(self, inputs: Tensor) -> Tensor:
        width = self.hidden_dim
        pooled = inputs[:, :2 * width].float().reshape(inputs.shape[0], 2, width)
        pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)
        normalized = torch.cat((pooled.reshape(inputs.shape[0], 2 * width),
                                inputs[:, 2 * width:].float()), -1)
        return self.selector(normalized)


class NormalizedFactorizedFlow(FactorizedFlow):
    """Only normalize pooled trunk scales before the original selector MLP.

    The factorized base, basis, selector weights, and selector's zero final
    projection keep their existing initialization and objective.
    """

    def __init__(self, config, base):
        super().__init__(config, base)
        self.selector = _NormalizedPooledSelector(self.selector, config.model.hidden_dim)
