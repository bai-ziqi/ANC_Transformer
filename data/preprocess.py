"""Reference-only log-PSD features; statistics are estimated on TRAIN alone."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LogSpectrum(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_fft = cfg["n_fft"]
        self.hop = cfg["hop_length"]
        self.bins = cfg["frequency_bins"]
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, waveform):
        if waveform.ndim != 2 or waveform.shape[-1] < self.n_fft:
            raise ValueError("Expected waveform [B,N] with N >= n_fft.")
        # center=False: no reflected future samples are created.
        value = torch.stft(waveform, self.n_fft, self.hop, window=self.window,
                           center=False, return_complex=True).abs().square()
        value = value / self.window.square().sum()
        value = F.adaptive_avg_pool1d(value.transpose(1, 2), self.bins)
        return torch.log(value.clamp_min(1e-12))  # [B, time_frames, feature_bins]
