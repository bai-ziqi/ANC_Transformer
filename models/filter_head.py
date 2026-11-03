"""Pool adapted time tokens and predict a residual around the Wiener prior."""
from __future__ import annotations

import torch
from torch import nn


class FilterHead(nn.Module):
    def __init__(self, width: int, taps: int):
        super().__init__()
        self.pool_score = nn.Linear(width, 1)
        self.residual = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, taps),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, decoded: torch.Tensor, prior_filter: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = self.pool_score(decoded).squeeze(-1)
        weights = scores.softmax(dim=-1)
        context = torch.einsum("bt,btc->bc", weights, decoded)
        adapted_filter = prior_filter + self.residual(context)
        return {"filter": adapted_filter, "pool_weights": weights}
