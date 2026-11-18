"""Pool adapted time tokens and generate a complete FIR directly."""
from __future__ import annotations

import torch
from torch import nn


class FilterHead(nn.Module):
    def __init__(
        self,
        width: int,
        taps: int,
        output_mode: str = "direct",
        scale_mode: str = "prior_rms",
        pooling_mode: str = "single_linear",
        num_queries: int = 1,
        num_heads: int = 4,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        return_pool_weights: bool = False,
    ):
        super().__init__()
        if output_mode not in ("direct", "residual"):
            raise ValueError("filter_output_mode must be direct or residual.")
        if scale_mode not in ("prior_rms", "unit"):
            raise ValueError("filter_scale_mode must be prior_rms or unit.")
        if pooling_mode not in ("single_linear", "multi_query_attention"):
            raise ValueError("Unknown filter_head pooling mode.")
        if width <= 0 or taps <= 0:
            raise ValueError("Filter head width and tap count must be positive.")
        if num_queries <= 0 or num_heads <= 0 or width % num_heads:
            raise ValueError("Invalid pooling query/head count.")
        self.width = width
        self.taps = taps
        self.output_mode = output_mode
        self.scale_mode = scale_mode
        self.pooling_mode = pooling_mode
        self.return_pool_weights = bool(return_pool_weights)

        if pooling_mode == "single_linear":
            self.pool_score = nn.Linear(width, 1)
            self.pool_queries = None
            self.pool_attention = None
            context_width = width
            head_width = width
            self.output = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                nn.Linear(width, taps),
            )
        else:
            self.pool_score = None
            self.pool_queries = nn.Parameter(torch.empty(1, num_queries, width))
            nn.init.normal_(self.pool_queries, std=0.02)
            self.pool_attention = nn.MultiheadAttention(
                embed_dim=width,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            context_width = num_queries * width
            head_width = width if hidden_dim is None else int(hidden_dim)
            if head_width <= 0:
                raise ValueError("Filter head hidden dimension must be positive.")
            self.output = nn.Sequential(
                nn.LayerNorm(context_width),
                nn.Linear(context_width, head_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_width, taps),
            )
        nn.init.zeros_(self.output[-1].bias)
        # Consume the same RNG path in both modes. Residual mode then zeroes
        # this layer, which makes w_hat == w_prior at initialization while
        # keeping seeded dropout masks comparable with the direct run.
        nn.init.xavier_uniform_(self.output[-1].weight)
        if output_mode == "residual":
            nn.init.zeros_(self.output[-1].weight)

    def forward(self, decoded: torch.Tensor, prior_filter: torch.Tensor) -> dict[str, torch.Tensor]:
        if decoded.ndim != 3 or decoded.shape[-1] != self.width or decoded.shape[1] == 0:
            raise ValueError(f"decoded must have shape [B,T,{self.width}] with T > 0.")
        if prior_filter.shape != (decoded.shape[0], self.taps):
            raise ValueError(f"Filter head prior_filter must have shape [B,{self.taps}].")
        if self.pooling_mode == "single_linear":
            scores = self.pool_score(decoded).squeeze(-1)
            weights = scores.softmax(dim=-1)
            context = torch.einsum("bt,btc->bc", weights, decoded)
        else:
            queries = self.pool_queries.expand(decoded.shape[0], -1, -1)
            pooled, weights = self.pool_attention(
                queries,
                decoded,
                decoded,
                need_weights=self.return_pool_weights,
                average_attn_weights=True,
            )
            context = pooled.flatten(1)
        generated = self.output(context)
        if self.output_mode == "direct":
            # The prior can supply coefficient scale, but the learned tokenizer
            # may also operate on the raw FIR when this is disabled.
            if self.scale_mode == "prior_rms":
                prior_scale = prior_filter.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            else:
                prior_scale = torch.ones_like(prior_filter[:, :1])
            adapted_filter = prior_scale * generated
        else:
            prior_scale = torch.ones_like(prior_filter[:, :1])
            adapted_filter = prior_filter + generated
        return {
            "filter": adapted_filter,
            "pool_weights": weights,
            "pooled_context": context,
            "prior_scale": prior_scale,
        }
