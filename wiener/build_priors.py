"""Build one independent broadband Wiener prior per acoustic environment."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from data.environments import load_environments
from utils import bank_signature_cfg, load_config
from wiener.build_targets import _atomic_json
from wiener.calculate_wiener import energy_diagnostics, fit_wiener_array


def _calibration_signal(settings, sample_rate, seconds, warmup, environment_id, stream):
    """Use named independent RNG streams; no source dataset clip is consulted."""
    low, high = float(settings["low_hz"]), float(settings["high_hz"])
    if not np.isfinite([low, high]).all() or not 0 < low < high < sample_rate / 2:
        raise ValueError("Prior calibration requires 0 < low_hz < high_hz < Nyquist.")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Prior calibration duration must be finite and positive.")
    order = settings.get("filter_order", 6)
    if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
        raise ValueError("prior.filter_order must be a positive integer.")
    seed = settings.get("seed", 2026)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("prior.seed must be a nonnegative integer.")
    name_seed = int.from_bytes(hashlib.sha256(environment_id.encode("utf-8")).digest()[:4], "little")
    entropy = [seed, name_seed, stream]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    count = round(seconds * sample_rate)
    if count <= 0:
        raise ValueError("Prior duration yields no samples.")
    design = butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    value = sosfilt(design, rng.standard_normal(warmup + count))
    rms = float(settings.get("rms", 0.1))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("prior.rms must be finite and positive.")
    power = float(np.mean(value[warmup:] ** 2))
    if not np.isfinite(power) or power <= 0:
        raise ValueError("Prior calibration signal has invalid power.")
    value *= rms / np.sqrt(power)
    return value, {"rng_entropy": entropy, "samples": count, "warmup_samples": warmup,
                   "bandpass": "causal_butterworth_sos", "filter_order": order,
                   "low_hz": low, "high_hz": high, "rms": rms}


def build_priors(cfg):
    """Fit broadband priors using separate synthetic fit and selection streams."""
    if cfg.get("protocol", {}).get("mode") != "sfanc":
        raise ValueError("build_priors requires protocol.mode=sfanc.")
    settings = cfg["prior"]
    sample_rate, taps = cfg["sample_rate"], cfg["filter_length"]
    environments = load_environments(cfg)
    warmup_seconds = float(settings.get("warmup_seconds", 0.25))
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("prior.warmup_seconds must be finite and nonnegative.")
    priors, primaries, secondaries, diagnostics = [], [], [], {}
    for environment_id, environment in environments.items():
        primary, secondary = environment["primary"], environment["secondary"]
        warmup = max(round(warmup_seconds * sample_rate), len(primary) - 1,
                     taps + len(secondary) - 2)
        fit_signal, fit_info = _calibration_signal(
            settings, sample_rate, float(settings["fit_seconds"]), warmup, environment_id, 0,
        )
        evaluation_signal, evaluation_info = _calibration_signal(
            settings, sample_rate, float(settings["validation_seconds"]), warmup, environment_id, 1,
        )
        coefficients, selection, rows = fit_wiener_array(
            fit_signal, primary, secondary, taps, cfg["wiener"]["ridge_ratios"],
            statistics_start=warmup, evaluation_reference=evaluation_signal,
            evaluation_start=warmup,
        )
        if not np.any(coefficients):
            raise ValueError(f"Environment {environment_id} produced an all-zero prior.")
        priors.append(coefficients)
        primaries.append(primary)
        secondaries.append(secondary)
        diagnostics[environment_id] = {
            "selection": selection, "candidate_ridges": rows,
            "fit_calibration": fit_info, "selection_calibration": evaluation_info,
            "path_provenance": environment["provenance"],
            "energy": energy_diagnostics(coefficients),
            "calibration_protocol": "independent synthetic broadband fit/selection streams; no dataset clips used",
        }
        print(f"Prior {environment_id}: rho={selection['ridge_ratio']:.6g}; "
              f"independent calibration MSE={selection['residual_mse']:.6e}", flush=True)
    environment_ids = list(environments)
    prior_array, primary_array, secondary_array = np.stack(priors), np.stack(primaries), np.stack(secondaries)
    bundle = {
        "format_version": 3,
        "protocol_mode": "sfanc",
        "environment_ids": environment_ids,
        "prior_filters": torch.from_numpy(prior_array),
        "primary_paths": torch.from_numpy(primary_array),
        "secondary_paths": torch.from_numpy(secondary_array),
        "sample_rate": sample_rate,
        "filter_length": taps,
        "primary_length": primary_array.shape[1],
        "secondary_length": secondary_array.shape[1],
        "config_signature": bank_signature_cfg(cfg),
        "diagnostics": diagnostics,
    }
    destination = Path(cfg["bank_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(destination)
    export = destination.with_suffix(".npz")
    temporary = export.with_name(export.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prior_filters=prior_array, primary_paths=primary_array,
                            secondary_paths=secondary_array, environment_ids=np.asarray(environment_ids),
                            sample_rate=np.asarray(sample_rate), filter_length=np.asarray(taps))
    temporary.replace(export)
    _atomic_json(destination.with_suffix(".json"), {
        "format_version": 3, "environment_ids": environment_ids, "sample_rate": sample_rate,
        "filter_length": taps, "diagnostics": diagnostics,
    })
    print(f"Saved environment priors {tuple(prior_array.shape)}: {destination}", flush=True)
    return bundle


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_priors(load_config(args.config))


if __name__ == "__main__":
    main()
