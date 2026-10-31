"""NumPy implementation of the existing project's unweighted third-octave scores.

Uses whole-signal mean removal, periodic Hann, right zero-padding, center=False,
one-sided power compensation, and nominal 1/3-octave centres. No A weighting.
"""
from __future__ import annotations

import numpy as np

CENTRES = np.asarray([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125,
                      160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                      1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                      10000, 12500, 16000, 20000], dtype=np.float64)


def power_spectrum(x, n_fft, hop):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    size = max(len(x), n_fft)
    size += (-(size - n_fft)) % hop
    x = np.pad(x, (0, size - len(x)))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    power = (np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2).mean(0) / np.dot(window, window)
    power[1:-1 if n_fft % 2 == 0 else None] *= 2
    return power


def score_signals(disturbance, residual, control, sample_rate=48000, n_fft=8192, hop_length=2048):
    d, e, y = [np.asarray(x, dtype=np.float64) for x in (disturbance, residual, control)]
    if d.ndim != 1 or d.shape != e.shape or d.shape != y.shape or not d.size:
        raise ValueError("Scores require matching nonempty 1-D signals.")
    if not all(np.isfinite(x).all() for x in (d, e, y)):
        raise ValueError("Nonfinite signals cannot be scored.")
    if n_fft <= 0 or not 0 < hop_length <= n_fft:
        raise ValueError("Invalid spectral window/hop.")
    pd, pe = [power_spectrum(x, n_fft, hop_length) for x in (d, e)]
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    factor = 2 ** (1 / 6)
    off, on, centres = [], [], []
    for centre in CENTRES:
        mask = (frequencies >= centre / factor) & (frequencies < centre * factor)
        if centre * factor <= sample_rate / 2 and mask.any():
            centres.append(centre)
            off.append(pd[mask].sum())
            on.append(pe[mask].sum())
    centres, off, on = map(np.asarray, (centres, off, on))
    nr = 10 * np.log10(np.maximum(off, 1e-30) / np.maximum(on, 1e-30))
    primary = (centres >= 50) & (centres <= 5000)
    high = (centres >= 1000) & (centres <= 8000)
    if not primary.any() or not high.any():
        raise ValueError("Not enough complete bands for primary/rebound scores.")
    worst = int(np.argmin(nr[high]))
    pd_time, pe_time, py_time = [float(np.mean(x ** 2)) for x in (d, e, y)]
    return {
        "primary_nr_db": float(nr[primary].mean()),
        "broadband_nr_db": float(10 * np.log10((np.sum(d ** 2) + 1e-20) / (np.sum(e ** 2) + 1e-20))),
        "total_band_nr_50_5000_db": float(10 * np.log10(max(off[primary].sum(), 1e-30) / max(on[primary].sum(), 1e-30))),
        "maximum_rebound_db": float(max(0, -nr[high].min())),
        "mean_positive_band_rebound_db": float(np.maximum(0, -nr[high]).mean()),
        "rebound_frequency_hz": float(centres[high][worst]),
        "disturbance_rms": float(np.sqrt(pd_time)),
        "residual_rms": float(np.sqrt(pe_time)),
        "control_rms": float(np.sqrt(py_time)),
        "control_peak_abs": float(np.abs(y).max()),
        "control_to_disturbance_db": float(10 * np.log10((py_time + 1e-20) / (pd_time + 1e-20))),
    }
