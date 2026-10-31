"""Reference-only log-PSD features; statistics are estimated on TRAIN alone."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.inspect_data import read_mono


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


def fit_normalizer(cfg, manifest):
    transform = LogSpectrum(cfg["features"])
    obs = round(cfg["features"]["observation_seconds"] * cfg["sample_rate"])
    count = cfg["features"]["normalization_windows_per_scene"]
    total, squared, frames = None, None, 0
    with torch.no_grad():
        for scene in manifest["scenes"]:
            a, b = scene["ranges"]["train"]
            for start in np.linspace(a, b - obs, count, dtype=np.int64):
                x = torch.tensor(read_mono(scene["sources"][0]["path"], int(start), obs), dtype=torch.float32)
                features = transform(x[None]).squeeze(0).double()
                s, q = features.sum(0), features.square().sum(0)
                total = s if total is None else total + s
                squared = q if squared is None else squared + q
                frames += len(features)
    mean = total / frames
    std = (squared / frames - mean.square()).clamp_min(1e-4).sqrt()
    return {"mean": mean.float(), "std": std.float(), "frames": frames,
            "source_split": "train", "windows_per_scene": count}
