"""The small tensor vocabulary shared by data, models, and sampling.

Pairs always use torch.triu_indices(N, N, offset=1) ordering. Padding is
storage only: masks exclude it from attention, noise, and every loss.
"""

from dataclasses import dataclass
import torch
from torch import Tensor

ATOM_TYPES = ("C", "N", "O", "F")
BOND_TYPES = ("none", "single", "double", "triple")


def pair_indices(max_nodes: int, device=None) -> Tensor:
    """Return [2,P] indices of unordered pairs, in one canonical order."""
    return torch.triu_indices(max_nodes, max_nodes, offset=1, device=device)


@dataclass(frozen=True)
class GraphMasks:
    nodes: Tensor
    pairs: Tensor

    @classmethod
    def from_counts(cls, counts: Tensor, max_nodes: int) -> "GraphMasks":
        if counts.ndim != 1 or torch.any((counts < 1) | (counts > max_nodes)):
            raise ValueError("counts must contain one value in [1,max_nodes] per graph")
        nodes = torch.arange(max_nodes, device=counts.device)[None] < counts[:, None]
        i, j = pair_indices(max_nodes, counts.device)
        return cls(nodes, nodes[:, i] & nodes[:, j])

    def to(self, device) -> "GraphMasks":
        return GraphMasks(self.nodes.to(device), self.pairs.to(device))


@dataclass(frozen=True)
class GraphBatch:
    atoms: Tensor
    bonds: Tensor
    masks: GraphMasks
    smiles: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()

    @property
    def counts(self) -> Tensor:
        return self.masks.nodes.sum(dim=1)

    def to(self, device) -> "GraphBatch":
        return GraphBatch(self.atoms.to(device), self.bonds.to(device),
                          self.masks.to(device), self.smiles, self.ids)


@dataclass(frozen=True)
class Latents:
    nodes: Tensor
    pairs: Tensor

    def masked(self, masks: GraphMasks) -> "Latents":
        return Latents(self.nodes * masks.nodes[..., None],
                       self.pairs * masks.pairs[..., None])

    def detach(self) -> "Latents":
        return Latents(self.nodes.detach(), self.pairs.detach())

    def to(self, device) -> "Latents":
        return Latents(self.nodes.to(device), self.pairs.to(device))


@dataclass(frozen=True)
class GraphLogits:
    atoms: Tensor
    bonds: Tensor


@dataclass(frozen=True)
class ModelConfig:
    max_nodes: int = 9
    node_dim: int = 64
    pair_dim: int = 32
    hidden_dim: int = 128
    depth: int = 4
    num_heads: int = 4

    def __post_init__(self):
        if min(self.max_nodes, self.node_dim, self.pair_dim, self.hidden_dim,
               self.depth, self.num_heads) < 1:
            raise ValueError("model dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
