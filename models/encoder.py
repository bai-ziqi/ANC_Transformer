"""Reference waveform frontends followed by RoPE-encoded Transformer layers."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from data.preprocess import LogSpectrum
from models.rope import RoPEEncoderLayer


class WaveformConvFrontend(nn.Module):
    """Convert a one-second waveform into coarse time tokens."""

    def __init__(self, feature_cfg, d_model: int):
        super().__init__()
        self.kernel_size = int(feature_cfg["conv_kernel_size"])
        self.causal = bool(feature_cfg.get("conv_causal", True))
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            stride=int(feature_cfg["conv_stride"]),
            padding=0 if self.causal else int(feature_cfg["conv_padding"]),
            bias=False,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(
            kernel_size=int(feature_cfg["conv_pool_kernel_size"]),
            stride=int(feature_cfg["conv_pool_stride"]),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.shape[1] < self.kernel_size:
            raise ValueError(
                f"Waveform frontend expects [B,N] with N >= {self.kernel_size}; "
                f"got {tuple(waveform.shape)}."
            )
        waveform = waveform.unsqueeze(1)  # [B,1,N]
        if self.causal:
            waveform = F.pad(waveform, (self.kernel_size - 1, 0))
        tokens = self.conv(waveform)
        tokens = self.activation(self.norm(tokens))
        tokens = self.pool(tokens)
        return tokens.transpose(1, 2)  # [B,T,d_model]


class NoiseEncoder(nn.Module):
    def __init__(self, feature_cfg, model_cfg, normalizer):
        super().__init__()
        width = model_cfg["d_model"]
        self.input_mode = feature_cfg.get("input_mode", "log_spectrum")
        self.register_buffer("feature_mean", None)
        self.register_buffer("feature_std", None)

        if self.input_mode == "waveform_conv":
            self.features = None
            self.projection = None
            self.waveform_frontend = WaveformConvFrontend(feature_cfg, width)
        elif self.input_mode == "log_spectrum":
            if not isinstance(normalizer, dict) or "mean" not in normalizer or "std" not in normalizer:
                raise ValueError("Log-spectrum input requires a feature normalizer.")
            self.features = LogSpectrum(feature_cfg)
            mean, std = normalizer["mean"], normalizer["std"]
            expected = (feature_cfg["frequency_bins"],)
            if (not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor)
                    or tuple(mean.shape) != expected or tuple(std.shape) != expected):
                raise ValueError(f"Feature normalizer mean/std must have shape {expected}.")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("Feature normalizer must be finite with strictly positive std.")
            self.feature_mean = mean.float().clone()
            self.feature_std = std.float().clone()
            self.projection = nn.Linear(feature_cfg["frequency_bins"], width)
            self.waveform_frontend = None
        else:
            raise ValueError(f"Unknown features.input_mode: {self.input_mode}.")

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
        if self.input_mode == "waveform_conv":
            tokens = self.waveform_frontend(reference_observation)
        else:
            features = (self.features(reference_observation) - self.feature_mean) / self.feature_std
            tokens = self.projection(features)
        # Bidirectional RoPE attention is confined to the already observed
        # window. No sinusoidal vector is added to the input embeddings.
        for layer in self.encoder:
            tokens = layer(tokens)
        return self.output_norm(tokens)
