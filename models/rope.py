"""Rotary position embeddings and a small Transformer encoder layer."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension by 90 degrees."""
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(value: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to [B, heads, time, head_dim] queries or keys."""
    if value.ndim != 4 or value.shape[-1] % 2:
        raise ValueError("RoPE expects [B,H,T,D] with an even D.")
    head_dim = value.shape[-1]
    positions = torch.arange(value.shape[-2], device=value.device, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32)
        * (-math.log(base) / head_dim)
    )
    angles = torch.outer(positions, frequencies)
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None]
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None]
    return value * cosine.to(value.dtype) + rotate_half(value) * sine.to(value.dtype)


class RoPESelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, rope_base: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads.")
        if (width // heads) % 2:
            raise ValueError("RoPE requires an even width per attention head.")
        self.heads = heads
        self.head_dim = width // heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = apply_rope(query, self.rope_base)
        key = apply_rope(key, self.rope_base)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        attended = weights @ value
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.dropout(self.output(attended))


class RoPEEncoderLayer(nn.Module):
    """Pre-norm self-attention layer with RoPE on temporal Q/K."""
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float, rope_base: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = RoPESelfAttention(width, heads, dropout, rope_base)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm_attention(tokens))
        tokens = tokens + self.dropout(self.feedforward(self.norm_feedforward(tokens)))
        return tokens
