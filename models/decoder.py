"""Wiener-prior-conditioned adaptation with Q from the reference encoder."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.rope import RoPESelfAttention


class WienerPriorTokenizer(nn.Module):
    """Represent one broadband FIR by ordered frequency-band tokens."""
    def __init__(self, taps: int, width: int, token_count: int):
        super().__init__()
        self.taps = taps
        self.tokens = token_count
        self.fft_size = 1 << (taps - 1).bit_length()
        self.projection = nn.Linear(2, width)
        self.frequency_embedding = nn.Parameter(torch.empty(1, token_count, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)

    def forward(self, prior_filter: torch.Tensor) -> torch.Tensor:
        if prior_filter.ndim != 2:
            raise ValueError("prior_filter must have shape [B,taps].")
        scale = prior_filter.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        spectrum = torch.fft.rfft(prior_filter / scale, n=self.fft_size)
        features = torch.stack((spectrum.real, spectrum.imag), dim=-1)
        features = F.adaptive_avg_pool1d(features.transpose(1, 2), self.tokens).transpose(1, 2)
        return self.projection(features) + self.frequency_embedding


class PriorCrossAttention(nn.Module):
    """Time query reads ordered Wiener frequency-band memory."""
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by nhead.")
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.attention_dropout = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, prior_tokens: torch.Tensor) -> torch.Tensor:
        batch, query_length, width = query_tokens.shape
        prior_length = prior_tokens.shape[1]
        query = self.query(query_tokens).view(batch, query_length, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(prior_tokens).view(batch, prior_length, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(prior_tokens).view(batch, prior_length, self.heads, self.head_dim).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        attended = (weights @ value).transpose(1, 2).reshape(batch, query_length, width)
        return self.dropout(self.output(attended))


class PriorDecoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float, rope_base: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.self_attention = RoPESelfAttention(width, heads, dropout, rope_base)
        self.norm_cross = nn.LayerNorm(width)
        self.cross_attention = PriorCrossAttention(width, heads, dropout)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, prior_tokens: torch.Tensor) -> torch.Tensor:
        query_tokens = query_tokens + self.self_attention(self.norm_self(query_tokens))
        query_tokens = query_tokens + self.cross_attention(self.norm_cross(query_tokens), prior_tokens)
        query_tokens = query_tokens + self.dropout(self.feedforward(self.norm_feedforward(query_tokens)))
        return query_tokens


class WienerDecoder(nn.Module):
    """A single broadband prior supplies K/V; encoder time tokens supply Q."""
    def __init__(self, taps: int, cfg):
        super().__init__()
        width = cfg["d_model"]
        self.prior_tokenizer = WienerPriorTokenizer(taps, width, cfg["prior_tokens"])
        self.layers = nn.ModuleList([
            PriorDecoderLayer(
                width,
                cfg["nhead"],
                cfg["dim_feedforward"],
                cfg["dropout"],
                cfg["rope_base"],
            )
            for _ in range(cfg["decoder_layers"])
        ])
        self.output_norm = nn.LayerNorm(width)

    def forward(self, encoder_queries: torch.Tensor, prior_filter: torch.Tensor) -> torch.Tensor:
        prior_tokens = self.prior_tokenizer(prior_filter)
        tokens = encoder_queries
        for layer in self.layers:
            tokens = layer(tokens, prior_tokens)
        return self.output_norm(tokens)
