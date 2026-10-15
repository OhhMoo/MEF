"""An ELF-derived graph encoder, clean-prediction flow, and terminal decoder.

One shared graph block updates unordered pairs symmetrically, collects their
messages at atoms, then applies ELF's position-free Transformer primitives.
All three components use this block with their own learned parameters.
"""

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .contracts import (ATOM_TYPES, BOND_TYPES, GraphBatch, GraphLogits, GraphMasks,
                        Latents, ModelConfig, pair_indices)
from .elf_layers import Attention, FinalLayer, RMSNorm, SwiGLUFFN, TimestepEmbedder, _make_linear


class GraphBlock(nn.Module):
    """Permutation-equivariant atom/pair communication without position labels."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.hidden_dim
        self.register_buffer("indices", pair_indices(config.max_nodes), persistent=False)
        self.pair_norm = RMSNorm(width)
        self.node_norm = RMSNorm(width)
        self.pair_update = nn.Sequential(_make_linear(3 * width, width), nn.SiLU(),
                                         _make_linear(width, width))
        self.incident_projection = _make_linear(width, width)
        self.attention_norm = RMSNorm(width)
        self.attention = Attention(width, config.num_heads)
        self.node_ffn_norm = RMSNorm(width)
        self.node_ffn = SwiGLUFFN(width, 4 * width)
        self.pair_ffn_norm = RMSNorm(width)
        self.pair_ffn = SwiGLUFFN(width, 4 * width)

    def forward(self, nodes: Tensor, pairs: Tensor, masks: GraphMasks) -> tuple[Tensor, Tensor]:
        i, j = self.indices
        h = self.node_norm(nodes)
        # A bond has no orientation: both endpoint features are symmetric in i,j.
        endpoints = torch.cat([self.pair_norm(pairs), h[:, i] + h[:, j],
                               (h[:, i] - h[:, j]).abs()], dim=-1)
        pairs = (pairs + self.pair_update(endpoints)) * masks.pairs[..., None]

        incident = torch.zeros_like(nodes)
        incident.index_add_(1, i, pairs)
        incident.index_add_(1, j, pairs)
        degree = nodes.new_zeros(masks.nodes.shape)
        degree.index_add_(1, i, masks.pairs.to(nodes.dtype))
        degree.index_add_(1, j, masks.pairs.to(nodes.dtype))
        nodes = nodes + self.incident_projection(incident / degree.clamp_min(1)[..., None])
        nodes = nodes + self.attention(self.attention_norm(nodes), masks.nodes)
        nodes = (nodes + self.node_ffn(self.node_ffn_norm(nodes))) * masks.nodes[..., None]
        pairs = (pairs + self.pair_ffn(self.pair_ffn_norm(pairs))) * masks.pairs[..., None]
        return nodes, pairs


class GraphBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList(GraphBlock(config) for _ in range(config.depth))

    def forward(self, nodes: Tensor, pairs: Tensor, masks: GraphMasks) -> tuple[Tensor, Tensor]:
        nodes, pairs = nodes * masks.nodes[..., None], pairs * masks.pairs[..., None]
        for block in self.blocks:
            nodes, pairs = block(nodes, pairs, masks)
        return nodes, pairs


class GraphEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.node_embedding = nn.Embedding(len(ATOM_TYPES), config.hidden_dim)
        self.pair_embedding = nn.Embedding(len(BOND_TYPES), config.hidden_dim)
        self.backbone = GraphBackbone(config)
        self.node_head = nn.Sequential(RMSNorm(config.hidden_dim),
                                       _make_linear(config.hidden_dim, config.node_dim))
        self.pair_head = nn.Sequential(RMSNorm(config.hidden_dim),
                                       _make_linear(config.hidden_dim, config.pair_dim))

    def forward(self, batch: GraphBatch) -> Latents:
        nodes, pairs = self.backbone(self.node_embedding(batch.atoms),
                                     self.pair_embedding(batch.bonds), batch.masks)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            return Latents(self.node_head(nodes.float()),
                           self.pair_head(pairs.float())).masked(batch.masks)


class GraphFlow(nn.Module):
    """Predict the clean normalized atom/pair destination F(z_t,t)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.node_input = _make_linear(config.node_dim, config.hidden_dim)
        self.pair_input = _make_linear(config.pair_dim, config.hidden_dim)
        self.time_embedding = TimestepEmbedder(config.hidden_dim)
        self.backbone = GraphBackbone(config)
        self.node_head = FinalLayer(config.hidden_dim, config.node_dim)
        self.pair_head = FinalLayer(config.hidden_dim, config.pair_dim)

    def forward(self, latents: Latents, t: Tensor, masks: GraphMasks) -> Latents:
        # ELF keeps projections/time conditioning and output heads in float32.
        with torch.autocast(device_type=latents.nodes.device.type, enabled=False):
            context = self.time_embedding(t.float())[:, None]
            nodes = self.node_input(latents.nodes.float()) + context
            pairs = self.pair_input(latents.pairs.float()) + context
        nodes, pairs = self.backbone(nodes, pairs, masks)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            return Latents(self.node_head(nodes.float()),
                           self.pair_head(pairs.float())).masked(masks)


class GraphDecoder(nn.Module):
    """Predict atom and unordered-pair categories only when requested."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.node_input = _make_linear(config.node_dim, config.hidden_dim)
        self.pair_input = _make_linear(config.pair_dim, config.hidden_dim)
        self.backbone = GraphBackbone(config)
        self.node_head = nn.Sequential(RMSNorm(config.hidden_dim),
                                       _make_linear(config.hidden_dim, len(ATOM_TYPES)))
        self.pair_head = nn.Sequential(RMSNorm(config.hidden_dim),
                                       _make_linear(config.hidden_dim, len(BOND_TYPES)))

    def forward(self, latents: Latents, masks: GraphMasks) -> GraphLogits:
        with torch.autocast(device_type=latents.nodes.device.type, enabled=False):
            nodes = self.node_input(latents.nodes.float())
            pairs = self.pair_input(latents.pairs.float())
        nodes, pairs = self.backbone(nodes, pairs, masks)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            return GraphLogits(self.node_head(nodes.float()) * masks.nodes[..., None],
                               self.pair_head(pairs.float()) * masks.pairs[..., None])


class LatentNormalizer(nn.Module):
    """Train-fitted channel statistics; identity until fit, never a learned map."""

    def __init__(self, node_dim: int, pair_dim: int):
        super().__init__()
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("pair_mean", torch.zeros(pair_dim))
        self.register_buffer("pair_std", torch.ones(pair_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, batches: Iterable[tuple[Latents, GraphMasks]]) -> None:
        totals = [torch.zeros_like(self.node_mean, device="cpu", dtype=torch.float64),
                  torch.zeros_like(self.pair_mean, device="cpu", dtype=torch.float64)]
        squares = [torch.zeros_like(total) for total in totals]
        counts = [0, 0]
        for latents, masks in batches:
            for k, (values, mask) in enumerate(((latents.nodes, masks.nodes),
                                                (latents.pairs, masks.pairs))):
                active = values.detach().cpu()[mask.detach().cpu()].double()
                totals[k] += active.sum(dim=0)
                squares[k] += active.square().sum(dim=0)
                counts[k] += active.shape[0]
        if not counts[0]:
            raise ValueError("normalizer fitting needs active training atoms")
        for k, (mean, std) in enumerate(((self.node_mean, self.node_std),
                                          (self.pair_mean, self.pair_std))):
            if counts[k]:
                fitted_mean = totals[k] / counts[k]
                variance = (squares[k] / counts[k] - fitted_mean.square()).clamp_min(0)
                mean.copy_(fitted_mean.to(mean))
                std.copy_(variance.sqrt().clamp_min(1e-4).to(std))
            else:
                mean.zero_()
                std.fill_(1)
        self.fitted.fill_(True)

    def forward(self, latents: Latents, masks: GraphMasks) -> Latents:
        return Latents((latents.nodes - self.node_mean) / self.node_std,
                       (latents.pairs - self.pair_mean) / self.pair_std).masked(masks)

    def inverse(self, latents: Latents, masks: GraphMasks) -> Latents:
        return Latents(latents.nodes * self.node_std + self.node_mean,
                       latents.pairs * self.pair_std + self.pair_mean).masked(masks)


class EMFModel(nn.Module):
    """One exploratory molecular generator with explicit E, F, and q modules."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GraphEncoder(config)
        self.flow = GraphFlow(config)
        self.decoder = GraphDecoder(config)
        self.normalizer = LatentNormalizer(config.node_dim, config.pair_dim)
        self._encoder_frozen = False
        self._decoder_frozen = False

    def encode(self, batch: GraphBatch) -> Latents:
        return self.normalizer(self.encoder(batch), batch.masks)

    def decode(self, latents: Latents, masks: GraphMasks) -> GraphLogits:
        return self.decoder(self.normalizer.inverse(latents, masks), masks)

    def freeze_encoder(self) -> None:
        self._encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.grad = None

    def freeze_decoder(self) -> None:
        """Hold q fixed while preserving derivatives with respect to its input."""
        self._decoder_frozen = True
        self.decoder.requires_grad_(False)
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        if self._decoder_frozen:
            self.decoder.eval()
        return self
