"""Reference waveform -> log-spectrum time tokens -> RoPE-encoded memory."""
from __future__ import annotations

import torch
from torch import nn

from data.preprocess import LogSpectrum
from models.rope import RoPEEncoderLayer


class NoiseEncoder(nn.Module):
    def __init__(self, feature_cfg, model_cfg, normalizer):
        super().__init__()
        self.features = LogSpectrum(feature_cfg)
        self.register_buffer("feature_mean", normalizer["mean"].clone())
        self.register_buffer("feature_std", normalizer["std"].clone())
        width = model_cfg["d_model"]
        self.projection = nn.Linear(feature_cfg["frequency_bins"], width)
        self.encoder = nn.ModuleList([
            RoPEEncoderLayer(
                width,
                model_cfg["nhead"],
                model_cfg["dim_feedforward"],
                model_cfg["dropout"],
                model_cfg["rope_base"],
            )
            for _ in range(model_cfg["encoder_layers"])
        ])
        self.output_norm = nn.LayerNorm(width)

    def forward(self, reference_observation):
        features = (self.features(reference_observation) - self.feature_mean) / self.feature_std
        tokens = self.projection(features)
        # Bidirectional RoPE attention is confined to the already observed
        # window. No sinusoidal vector is added to the input embeddings.
        for layer in self.encoder:
            tokens = layer(tokens)
        return self.output_norm(tokens)
