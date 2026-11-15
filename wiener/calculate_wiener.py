"""Biased stationary Toeplitz statistics and ridge-calibrated causal FIR solves."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate

from anc.anc_simulator import causal_numpy, simulate_numpy


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


def energy_diagnostics(coefficients):
    energy = np.cumsum(np.square(coefficients, dtype=np.float64))
    if energy[-1] <= 0 or not np.isfinite(energy[-1]):
        raise ValueError("Cannot form energy diagnostic for zero/nonfinite FIR.")
    energy /= energy[-1]
    return {"taps_for_95_percent": int(np.searchsorted(energy, 0.95) + 1),
            "taps_for_99_percent": int(np.searchsorted(energy, 0.99) + 1),
            "coefficient_l2": float(np.linalg.norm(coefficients)),
            "coefficient_peak_abs": float(np.abs(coefficients).max())}


def fit_wiener_array(reference, primary, secondary, taps, ridges, *,
                     statistics_start=0, evaluation_reference=None, evaluation_start=0):
    """Fit Toeplitz candidates and select by actual causal residual MSE.

    This is a candidate-grid reference, not an exact finite-window least-squares
    optimum. Each clip starts from zero filter state. ``statistics_start`` and
    ``evaluation_start`` may discard a calibration warmup without resetting any
    intermediate filter state.
    """
    vectors = []
    for name, value in (("reference", reference), ("primary", primary),
                        ("secondary", secondary)):
        value = np.asarray(value)
        if (value.ndim != 1 or not value.size or np.iscomplexobj(value)
                or not np.isfinite(value).all()):
            raise ValueError(f"{name} must be a nonempty finite real vector.")
        vectors.append(np.asarray(value, dtype=np.float64))
    reference, primary, secondary = vectors
    if not np.any(primary) or not np.any(secondary):
        raise ValueError("Primary and secondary paths must have nonzero energy.")
    if not isinstance(taps, (int, np.integer)) or isinstance(taps, bool) or taps <= 0:
        raise ValueError("taps must be a positive integer.")
    if not isinstance(statistics_start, (int, np.integer)) or not 0 <= statistics_start < len(reference):
        raise ValueError("Invalid statistics_start.")
    ratios = [float(value) for value in ridges]
    if not ratios or any(not np.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("Ridge ratios must be finite and nonnegative.")
    if len(set(ratios)) != len(ratios):
        raise ValueError("Ridge ratios must be distinct.")
    disturbance = causal_numpy(reference, primary)
    filtered_reference = causal_numpy(reference, secondary)
    auto, cross = statistic_vectors(filtered_reference[statistics_start:],
                                    disturbance[statistics_start:], taps)
    candidates = solve_candidates(auto, cross, ratios)
    if evaluation_reference is None:
        evaluation_reference = reference
        evaluation_disturbance = disturbance
    else:
        evaluation_reference = np.asarray(evaluation_reference)
        if (evaluation_reference.ndim != 1 or not evaluation_reference.size
                or np.iscomplexobj(evaluation_reference)
                or not np.isfinite(evaluation_reference).all()):
            raise ValueError("evaluation_reference must be a finite real vector.")
        evaluation_reference = np.asarray(evaluation_reference, dtype=np.float64)
        evaluation_disturbance = causal_numpy(evaluation_reference, primary)
    if (not isinstance(evaluation_start, (int, np.integer))
            or not 0 <= evaluation_start < len(evaluation_reference)):
        raise ValueError("Invalid evaluation_start.")
    rows = []
    for ratio, coefficients in candidates.items():
        outcome = simulate_numpy(evaluation_reference, evaluation_disturbance,
                                 coefficients, secondary, context=0)
        residual = outcome["residual"][evaluation_start:]
        control = outcome["control"][evaluation_start:]
        with np.errstate(over="ignore", invalid="ignore"):
            row = {
                "ridge_ratio": float(ratio),
                "absolute_ridge": float(ratio * auto[0]),
                "residual_mse": float(np.mean(np.square(residual))),
                "control_rms": float(np.sqrt(np.mean(np.square(control)))),
                "coefficient_l2": float(np.linalg.norm(coefficients)),
                "coefficient_peak_abs": float(np.abs(coefficients).max()),
            }
        if np.isfinite(list(row.values())).all():
            rows.append(row)
    if not rows:
        raise ValueError("All Wiener candidates produced nonfinite residuals.")
    # Deterministic ties prefer lower coefficient energy, then lower ridge.
    selected = min(rows, key=lambda row: (row["residual_mse"],
                                          row["coefficient_l2"], row["ridge_ratio"]))
    selection = dict(selected)
    selection.update({
        "selection_metric": "minimum_causal_residual_mse",
        "solver": "biased_stationary_toeplitz",
        "reference_kind": "best_in_configured_ridge_candidate_grid",
        "filtered_reference_power": float(auto[0]),
        "statistics_samples": len(reference) - int(statistics_start),
        "scored_samples": len(evaluation_reference) - int(evaluation_start),
        "failed_ridge_ratios": [value for value in ratios
                                if value not in {row["ridge_ratio"] for row in rows}],
    })
    return candidates[selected["ridge_ratio"]], selection, rows
