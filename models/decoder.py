"""Wiener-prior-conditioned adaptation with Q from the reference encoder."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.rope import RoPESelfAttention


class WienerPriorTokenizer(nn.Module):
    """Represent one environment FIR prior as learned or pooled FFT tokens."""
    SUPPORTED_MODES = ("linear", "mlp", "fft_pool")

    def __init__(
        self,
        taps: int,
        width: int,
        token_count: int,
        mode: str = "linear",
        normalize_input: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        add_frequency_embedding: bool = True,
    ):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"prior_tokenizer must be one of {self.SUPPORTED_MODES}; got {mode}."
            )
        if token_count <= 0 or width <= 0:
            raise ValueError("token_count and width must be positive.")
        if mode == "mlp" and hidden_dim <= 0:
            raise ValueError("prior_hidden_dim must be positive for MLP tokenization.")
        self.taps = taps
        self.width = width
        self.tokens = token_count
        self.mode = mode
        self.normalize_input = normalize_input
        output_dim = token_count * width
        if mode == "linear":
            # Learn the FIR-to-token decomposition directly instead of fixing it
            # to an FFT followed by average pooling.
            self.projection = nn.Linear(taps, output_dim)
        elif mode == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(taps, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.fft_size = 1 << (taps - 1).bit_length()
            self.projection = nn.Linear(2, width)
        if add_frequency_embedding:
            self.frequency_embedding = nn.Parameter(torch.empty(1, token_count, width))
            nn.init.normal_(self.frequency_embedding, std=0.02)
        else:
            self.register_parameter("frequency_embedding", None)

    def forward(self, prior_filter: torch.Tensor) -> torch.Tensor:
        if prior_filter.ndim != 2:
            raise ValueError("prior_filter must have shape [B,taps].")
        if prior_filter.shape[1] != self.taps:
            raise ValueError(f"prior_filter must have {self.taps} taps.")
        source = prior_filter
        if self.normalize_input:
            scale = source.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            source = source / scale
        if self.mode == "fft_pool":
            spectrum = torch.fft.rfft(source, n=self.fft_size)
            features = torch.stack((spectrum.real, spectrum.imag), dim=-1)
            features = F.adaptive_avg_pool1d(
                features.transpose(1, 2), self.tokens
            ).transpose(1, 2)
            tokens = self.projection(features)
        else:
            tokens = self.projection(source).reshape(
                prior_filter.shape[0], self.tokens, self.width
            )
        if self.frequency_embedding is not None:
            tokens = tokens + self.frequency_embedding
        return tokens


class PriorCrossAttention(nn.Module):
    """Encoder time queries read the environment's Wiener-prior tokens."""
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
        self.prior_tokenizer = WienerPriorTokenizer(
            taps,
            width,
            cfg["prior_tokens"],
            mode=cfg.get("prior_tokenizer", "linear"),
            normalize_input=cfg.get("prior_normalize", False),
            hidden_dim=cfg.get("prior_hidden_dim", 256),
            dropout=cfg.get("dropout", 0.0),
            add_frequency_embedding=cfg.get("prior_frequency_embedding", True),
        )
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
