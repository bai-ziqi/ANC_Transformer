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
    ):
        super().__init__()
        if output_mode not in ("direct", "residual"):
            raise ValueError("filter_output_mode must be direct or residual.")
        if scale_mode not in ("prior_rms", "unit"):
            raise ValueError("filter_scale_mode must be prior_rms or unit.")
        if width <= 0 or taps <= 0:
            raise ValueError("Filter head width and tap count must be positive.")
        self.width = width
        self.taps = taps
        self.output_mode = output_mode
        self.scale_mode = scale_mode
        self.pool_score = nn.Linear(width, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, taps),
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
        scores = self.pool_score(decoded).squeeze(-1)
        weights = scores.softmax(dim=-1)
        context = torch.einsum("bt,btc->bc", weights, decoded)
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
        return {"filter": adapted_filter, "pool_weights": weights, "prior_scale": prior_scale}
