"""Causal finite-episode ANC: e = d - s * (w * x), with explicit state timing."""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import fftconvolve


def causal_numpy(signal, kernel):
    return fftconvolve(signal, kernel, mode="full")[:len(signal)]


def simulate_numpy(reference, disturbance, coefficients, secondary, context=0):
    reference, disturbance = np.asarray(reference), np.asarray(disturbance)
    if reference.ndim != 1 or reference.shape != disturbance.shape or not 0 <= context < len(reference):
        raise ValueError("Need matching nonempty 1-D audio and valid context.")
    y = causal_numpy(reference, coefficients)
    # No actuator output existed before decision time. Only reference FIR
    # history is prefilled. All future samples, including startup, are scored.
    y[:context] = 0.0
    anti = causal_numpy(y, secondary)
    return {"control": y[context:], "anti_noise": anti[context:],
            "residual": disturbance[context:] - anti[context:]}


def causal_fir_batch(signal, kernels):
    if signal.ndim != 2:
        raise ValueError("signal must be [B,N].")
    if kernels.ndim == 1:
        kernels = kernels[None].expand(signal.shape[0], -1)
    if kernels.ndim != 2 or kernels.shape[0] != signal.shape[0]:
        raise ValueError("kernels must be [M] or [B,M].")
    full = signal.shape[-1] + kernels.shape[-1] - 1
    n_fft = 1 << (full - 1).bit_length()
    result = torch.fft.irfft(torch.fft.rfft(signal, n=n_fft) *
                             torch.fft.rfft(kernels, n=n_fft), n=n_fft)
    return result[:, :signal.shape[-1]]


def simulate(reference, disturbance, coefficients, secondary, context=0):
    if reference.shape != disturbance.shape or not 0 <= context < reference.shape[-1]:
        raise ValueError("Audio shapes/context do not match.")
    y = causal_fir_batch(reference, coefficients)
    y = torch.cat((torch.zeros_like(y[:, :context]), y[:, context:]), dim=-1)
    anti = causal_fir_batch(y, secondary)
    return {"control": y[:, context:], "anti_noise": anti[:, context:],
            "residual": disturbance[:, context:] - anti[:, context:]}
