"""Pool adapted time tokens and generate a complete FIR directly."""
from __future__ import annotations

import torch
from torch import nn


class FilterHead(nn.Module):
    def __init__(self, width: int, taps: int, output_mode: str = "direct"):
        super().__init__()
        if output_mode not in ("direct", "residual"):
            raise ValueError("filter_output_mode must be direct or residual.")
        self.output_mode = output_mode
        self.pool_score = nn.Linear(width, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, taps),
        )
        nn.init.zeros_(self.output[-1].bias)
        if output_mode == "residual":
            nn.init.zeros_(self.output[-1].weight)
        else:
            nn.init.xavier_uniform_(self.output[-1].weight)

    def forward(self, decoded: torch.Tensor, prior_filter: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = self.pool_score(decoded).squeeze(-1)
        weights = scores.softmax(dim=-1)
        context = torch.einsum("bt,btc->bc", weights, decoded)
        generated = self.output(context)
        if self.output_mode == "direct":
            # The decoder generates every normalized FIR coefficient. The
            # prior only supplies the environment-level coefficient scale.
            prior_scale = prior_filter.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            adapted_filter = prior_scale * generated
        else:
            prior_scale = torch.ones_like(prior_filter[:, :1])
            adapted_filter = prior_filter + generated
        return {"filter": adapted_filter, "pool_weights": weights, "prior_scale": prior_scale}
