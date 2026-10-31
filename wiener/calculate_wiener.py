"""Biased stationary Toeplitz statistics and ridge-calibrated causal FIR solves."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate

from anc.anc_simulator import causal_numpy, simulate_numpy
from anc.metrics import score_signals
from data.inspect_data import read_mono


def statistic_vectors(z, d, taps):
    """1/N biased autocorrelation (PSD Toeplitz) and E[d[t]z[t-k]]."""
    if len(z) != len(d) or len(z) <= taps:
        raise ValueError("Wiener segments must match and exceed FIR length.")
    start = len(z) - 1
    auto = correlate(z, z, mode="full", method="fft")[start:start + taps] / len(z)
    cross = correlate(d, z, mode="full", method="fft")[start:start + taps] / len(z)
    if not np.isfinite(auto).all() or not np.isfinite(cross).all() or auto[0] <= 0:
        raise ValueError("Invalid/zero-power Wiener statistics.")
    return auto, cross


def collect_statistics(cfg, scene, secondary):
    sr, taps = cfg["sample_rate"], cfg["filter_length"]
    a, b = scene["ranges"]["wiener_fit"]
    context = 2 * (taps - 1)
    segments = cfg["wiener"]["fit_segments"]
    n = round(cfg["wiener"]["fit_seconds_per_scene"] * sr / segments)
    if n <= taps or segments * n > b - a - context:
        raise ValueError("Wiener fit duration does not fit training-only interval.")
    starts = np.linspace(a + context, b - n, segments, dtype=np.int64)
    if segments > 1 and np.min(np.diff(starts)) < n:
        raise ValueError("Wiener fit segments overlap; decrease fit duration.")
    auto, cross, intervals = np.zeros(taps), np.zeros(taps), []
    for start in starts:
        start = int(start)
        read_start = start - context
        assert a <= read_start and start + n <= b
        x = read_mono(scene["sources"][0]["path"], read_start, context + n)
        d = read_mono(scene["sources"][1]["path"], read_start, context + n)
        z = causal_numpy(x, secondary)[context:]
        r, p = statistic_vectors(z, d[context:], taps)
        auto += r / segments
        cross += p / segments
        intervals.append({"read": [read_start, start + n], "statistics": [start, start + n]})
    return auto, cross, {"source_split": "train", "intervals": intervals, "fit_samples": n * segments}


def solve_candidates(auto, cross, ridges):
    candidates = {}
    for ridge in ridges:
        column = auto.copy()
        column[0] += ridge * auto[0]
        try:
            coefficients = solve_toeplitz((column, column), cross, check_finite=True)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(coefficients).all():
            candidates[float(ridge)] = coefficients
    if not candidates:
        raise ValueError("All Wiener solves failed.")
    return candidates


def calibration_audio(scene, cfg):
    a, b = scene["ranges"]["wiener_calibration"]
    x = read_mono(scene["sources"][0]["path"], a, b - a)
    d = read_mono(scene["sources"][1]["path"], a, b - a)
    return x, d


def choose_candidate(candidates, audio_pairs, secondary, cfg):
    """All audio_pairs originate in INNER TRAIN, never outer validation/test."""
    rows = []
    context = cfg["filter_length"] - 1
    for ridge, coefficients in candidates.items():
        metrics = []
        for x, d in audio_pairs:
            result = simulate_numpy(x, d, coefficients, secondary, context)
            metrics.append(score_signals(d[context:], result["residual"], result["control"],
                                         cfg["sample_rate"], **cfg["metrics"]))
        rows.append({"ridge_ratio": ridge,
                     "mean_primary_nr_db": float(np.mean([m["primary_nr_db"] for m in metrics])),
                     "maximum_rebound_db": float(max(m["maximum_rebound_db"] for m in metrics)),
                     "coefficient_l2": float(np.linalg.norm(coefficients))})
    selected = max(rows, key=lambda r: (r["mean_primary_nr_db"], -r["maximum_rebound_db"]))
    return candidates[selected["ridge_ratio"]], selected, rows


def energy_diagnostics(coefficients):
    energy = np.cumsum(np.square(coefficients, dtype=np.float64))
    if energy[-1] <= 0 or not np.isfinite(energy[-1]):
        raise ValueError("Cannot form energy diagnostic for zero/nonfinite FIR.")
    energy /= energy[-1]
    return {"taps_for_95_percent": int(np.searchsorted(energy, 0.95) + 1),
            "taps_for_99_percent": int(np.searchsorted(energy, 0.99) + 1),
            "coefficient_l2": float(np.linalg.norm(coefficients)),
            "coefficient_peak_abs": float(np.abs(coefficients).max())}
