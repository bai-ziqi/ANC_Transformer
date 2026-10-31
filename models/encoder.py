"""Reference waveform -> log-spectrum time sequence -> Transformer memory."""
from __future__ import annotations

import math

import torch
from torch import nn

from data.preprocess import LogSpectrum


def positional_encoding(length, width, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    scales = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales[:width // 2])
    return encoding


class NoiseEncoder(nn.Module):
    def __init__(self, feature_cfg, model_cfg, normalizer):
        super().__init__()
        self.features = LogSpectrum(feature_cfg)
        self.register_buffer("feature_mean", normalizer["mean"].clone())
        self.register_buffer("feature_std", normalizer["std"].clone())
        width = model_cfg["d_model"]
        self.projection = nn.Linear(feature_cfg["frequency_bins"], width)
        layer = nn.TransformerEncoderLayer(width, model_cfg["nhead"], model_cfg["dim_feedforward"],
                                           model_cfg["dropout"], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, model_cfg["encoder_layers"],
                                             norm=nn.LayerNorm(width), enable_nested_tensor=False)

    def forward(self, reference_observation):
        features = (self.features(reference_observation) - self.feature_mean) / self.feature_std
        tokens = self.projection(features)
        tokens = tokens + positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
        # Bidirectional attention is confined to the ALREADY OBSERVED window;
        # the scored future interval never enters this encoder.
        return self.encoder(tokens)
