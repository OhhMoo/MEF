"""Small ELF PyTorch primitives, adapted for position-free molecular graphs.

Source: 0_external_repo/ELF_pytorch/src/modules/layers.py, commit
b29d8833609e9ab7f67cd9da39435ac5cea04837. RMSNorm, timestep embedding,
SwiGLU, and the clean-prediction head retain the upstream implementation.
Attention omits text rotary positions and dropout. No tokenizer is needed.

MIT License
Copyright (c) 2026 ELF authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _make_linear(in_features, out_features, bias=True, kernel_init=nn.init.xavier_uniform_,
                 bias_init=nn.init.zeros_):
    layer = nn.Linear(in_features, out_features, bias=bias)
    kernel_init(layer.weight)
    if bias and bias_init is not None:
        bias_init(layer.bias)
    return layer


def _normal_init_002(tensor):
    nn.init.normal_(tensor, mean=0.0, std=0.02)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + self.eps).to(input_dtype)
        return self.weight.to(input_dtype) * (hidden_states * inv_std)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = _make_linear(frequency_embedding_size, hidden_size,
                                  kernel_init=_normal_init_002)
        self.mlp_2 = _make_linear(hidden_size, hidden_size, kernel_init=_normal_init_002)

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_emb = self.mlp_0(self.timestep_embedding(t, self.frequency_embedding_size))
        return self.mlp_2(F.silu(t_emb))


class Attention(nn.Module):
    """ELF Q/K-normalized attention, with no atom-index position encoding."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = _make_linear(dim, dim * 3)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        self.proj = _make_linear(dim, dim)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        mask = attention_mask[:, None, None, :].bool()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.permute(0, 2, 1, 3).reshape(batch, length, dim))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim_eff = int(hidden_dim * 2 / 3)
        self.w12 = _make_linear(dim, 2 * hidden_dim_eff)
        self.w3 = _make_linear(hidden_dim_eff, dim)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class FinalLayer(nn.Module):
    """ELF's zero-initialized clean-destination prediction head."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = _make_linear(hidden_size, out_channels,
                                   kernel_init=nn.init.zeros_, bias_init=nn.init.zeros_)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm_final(x))
