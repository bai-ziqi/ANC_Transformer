"""Reference-query / Wiener-prior-conditioned adaptive ANC filter generator."""
from __future__ import annotations

import torch
from torch import nn

from models.decoder import WienerDecoder
from models.encoder import NoiseEncoder
from models.filter_head import FilterHead


class ANCTransformer(nn.Module):
    def __init__(self, cfg, bundle):
        super().__init__()
        self.architecture = cfg["model"]["architecture"]
        if self.architecture != "prior_adaptive":
            raise ValueError(f"Unknown model architecture: {self.architecture}")
        self.sample_rate = cfg["sample_rate"]
        self.filter_length = cfg["filter_length"]
        self.primary_length = cfg.get("primary_length")
        self.secondary_length = cfg.get("secondary_length")
        if bundle.get("format_version") == 3:
            for key in ("sample_rate", "filter_length", "primary_length", "secondary_length"):
                if bundle.get(key) != cfg.get(key):
                    raise ValueError(f"Environment bundle differs from model config at {key}.")
            environments = len(bundle.get("environment_ids", []))
            if not environments:
                raise ValueError("Environment bundle requires nonempty environment_ids.")
            # Physical path lengths do not define the prior or output FIR length.
            for key, length in (("primary_paths", self.primary_length),
                                ("secondary_paths", self.secondary_length),
                                ("prior_filters", self.filter_length)):
                value = bundle.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != (environments, length):
                    raise ValueError(f"{key} must have shape [{environments}, {length}].")
        self.encoder = NoiseEncoder(cfg["features"], cfg["model"], bundle["normalizer"])
        self.decoder = WienerDecoder(self.filter_length, cfg["model"])
        self.filter_head = FilterHead(
            cfg["model"]["d_model"],
            self.filter_length,
            cfg["model"].get("filter_output_mode", "direct"),
            cfg["model"].get("filter_scale_mode", "prior_rms"),
            pooling_mode=cfg["model"].get("pooling_mode", "single_linear"),
            num_queries=cfg["model"].get("pool_queries", 1),
            num_heads=cfg["model"]["nhead"],
            hidden_dim=cfg["model"].get("filter_head_hidden", cfg["model"]["d_model"]),
            dropout=cfg["model"]["dropout"],
            return_pool_weights=cfg["model"].get("return_pool_weights", False),
        )

    def forward(self, reference_observation, prior_filter=None):
        if (not isinstance(reference_observation, torch.Tensor)
                or reference_observation.ndim != 2 or min(reference_observation.shape) <= 0
                or not torch.is_floating_point(reference_observation)):
            raise ValueError("reference_observation must be a nonempty floating-point tensor [B,N].")
        parameter = next(self.parameters())
        reference_observation = reference_observation.to(device=parameter.device, dtype=parameter.dtype)
        batch = reference_observation.shape[0]
        if prior_filter is None:
            raise ValueError("prior_filter [taps] or [B,taps] is required for prior_adaptive.")
        else:
            prior_filter = torch.as_tensor(prior_filter, device=reference_observation.device,
                                           dtype=reference_observation.dtype)
            if prior_filter.ndim == 1:
                prior_filter = prior_filter[None].expand(batch, -1)
            if prior_filter.shape != (batch, self.filter_length):
                raise ValueError(
                    f"prior_filter must have shape [{self.filter_length}] or "
                    f"[B,{self.filter_length}]; got {tuple(prior_filter.shape)}."
                )
        encoder_queries = self.encoder(reference_observation)
        decoded = self.decoder(encoder_queries, prior_filter)
        result = self.filter_head(decoded, prior_filter)
        result["prior_filter"] = prior_filter
        return result
