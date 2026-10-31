"""Learn 8 logits, then form a convex combination of the FROZEN raw bank."""
from __future__ import annotations

from torch import nn


class FilterHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, decoded, bank):
        logits = self.score(decoded).squeeze(-1)
        weights = logits.softmax(dim=-1)
        return {"logits": logits, "weights": weights, "filter": weights @ bank}
