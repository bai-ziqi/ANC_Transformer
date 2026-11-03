"""Legacy eight-candidate Bank-selection transformer kept for reproducibility."""
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


class LegacyEncoder(nn.Module):
    def __init__(self, feature_cfg, model_cfg, normalizer):
        super().__init__()
        self.features = LogSpectrum(feature_cfg)
        self.register_buffer("feature_mean", normalizer["mean"].clone())
        self.register_buffer("feature_std", normalizer["std"].clone())
        width = model_cfg["d_model"]
        self.projection = nn.Linear(feature_cfg["frequency_bins"], width)
        layer = nn.TransformerEncoderLayer(
            width, model_cfg["nhead"], model_cfg["dim_feedforward"],
            model_cfg["dropout"], batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, model_cfg["encoder_layers"], norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )

    def forward(self, reference_observation):
        features = (self.features(reference_observation) - self.feature_mean) / self.feature_std
        tokens = self.projection(features)
        tokens = tokens + positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
        return self.encoder(tokens)


class LegacyDecoder(nn.Module):
    def __init__(self, taps, scene_count, cfg):
        super().__init__()
        width = cfg["d_model"]
        self.projection = nn.Linear(taps, width)
        self.scene_embedding = nn.Embedding(scene_count, width)
        layer = nn.TransformerDecoderLayer(
            width, cfg["nhead"], cfg["dim_feedforward"],
            cfg["dropout"], batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg["decoder_layers"], norm=nn.LayerNorm(width))

    def forward(self, bank, memory):
        identities = torch.arange(len(bank), device=bank.device)
        scale = bank.square().mean().sqrt().clamp_min(1e-8)
        tokens = self.projection(bank / scale) + self.scene_embedding(identities)
        queries = tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
        return self.decoder(tgt=queries, memory=memory)


class LegacyFilterHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, decoded, bank):
        logits = self.score(decoded).squeeze(-1)
        weights = logits.softmax(dim=-1)
        return {"logits": logits, "weights": weights, "filter": weights @ bank}
