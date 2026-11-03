"""Reference-query / Wiener-prior-conditioned adaptive ANC filter generator."""
from __future__ import annotations

import torch
from torch import nn

from models.decoder import WienerDecoder
from models.encoder import NoiseEncoder
from models.filter_head import FilterHead
from models.legacy import LegacyDecoder, LegacyEncoder, LegacyFilterHead


class ANCTransformer(nn.Module):
    def __init__(self, cfg, bundle):
        super().__init__()
        self.architecture = cfg["model"]["architecture"]
        if self.architecture == "prior_adaptive":
            self.encoder = NoiseEncoder(cfg["features"], cfg["model"], bundle["normalizer"])
            self.decoder = WienerDecoder(cfg["filter_length"], cfg["model"])
            self.filter_head = FilterHead(cfg["model"]["d_model"], cfg["filter_length"])
        elif self.architecture == "legacy_bank_fusion":
            self.register_buffer("filter_bank", bundle["filters"].float().clone())
            self.encoder = LegacyEncoder(cfg["features"], cfg["model"], bundle["normalizer"])
            self.decoder = LegacyDecoder(cfg["filter_length"], len(cfg["scene_names"]), cfg["model"])
            self.filter_head = LegacyFilterHead(cfg["model"]["d_model"])
        else:
            raise ValueError(f"Unknown model architecture: {self.architecture}")

    def forward(self, reference_observation, prior_filter=None):
        if self.architecture == "legacy_bank_fusion":
            memory = self.encoder(reference_observation)
            decoded = self.decoder(self.filter_bank, memory)
            return self.filter_head(decoded, self.filter_bank)

        batch = reference_observation.shape[0]
        if prior_filter is None:
            raise ValueError("prior_filter [taps] or [B,taps] is required for prior_adaptive.")
        else:
            prior_filter = torch.as_tensor(prior_filter, device=reference_observation.device,
                                           dtype=reference_observation.dtype)
            if prior_filter.ndim == 1:
                prior_filter = prior_filter[None].expand(batch, -1)
            if prior_filter.shape != (batch, self.decoder.prior_tokenizer.taps):
                raise ValueError("prior_filter must have shape [taps] or [B,taps].")
        encoder_queries = self.encoder(reference_observation)
        decoded = self.decoder(encoder_queries, prior_filter)
        result = self.filter_head(decoded, prior_filter)
        result["prior_filter"] = prior_filter
        return result
