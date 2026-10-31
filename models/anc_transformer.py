"""Single-path, reference-only, bank-conditioned ANC filter selection/fusion."""
from __future__ import annotations

from torch import nn

from models.decoder import WienerDecoder
from models.encoder import NoiseEncoder
from models.filter_head import FilterHead


class ANCTransformer(nn.Module):
    def __init__(self, cfg, bundle):
        super().__init__()
        self.register_buffer("filter_bank", bundle["filters"].float().clone())
        self.encoder = NoiseEncoder(cfg["features"], cfg["model"], bundle["normalizer"])
        self.decoder = WienerDecoder(cfg["filter_length"], len(cfg["scene_names"]), cfg["model"])
        self.filter_head = FilterHead(cfg["model"]["d_model"])

    def forward(self, reference_observation):
        memory = self.encoder(reference_observation)
        decoded = self.decoder(self.filter_bank, memory)
        return self.filter_head(decoded, self.filter_bank)
