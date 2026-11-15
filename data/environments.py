"""Read explicit acoustic environments without changing gain or propagation delay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import loadmat

from utils import sha256


_RATE_KEYS = ("fs", "Fs", "FS", "sample_rate", "samplerate", "sampling_rate")


def _real_vector(value, label):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or np.iscomplexobj(value):
        raise ValueError(f"{label} must be a real numeric vector.")
    value = value.squeeze()
    if value.ndim != 1 or value.size < 2:
        raise ValueError(f"{label} must contain one vector, not a scalar or a matrix.")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains nonfinite values.")
    return value


def _mat_rate(values, label):
    rates = []
    for key in _RATE_KEYS:
        if key not in values:
            continue
        value = np.asarray(values[key]).squeeze()
        if value.ndim or value.dtype.kind not in "fiu" or not np.isfinite(value):
            raise ValueError(f"{label}: {key} must be a finite scalar sample rate.")
        rate = float(value)
        if rate <= 0 or not rate.is_integer():
            raise ValueError(f"{label}: {key} must be a positive integer sample rate.")
        rates.append(int(rate))
    if len(set(rates)) > 1:
        raise ValueError(f"{label} contains inconsistent sample-rate metadata.")
    return rates[0] if rates else None


def read_vector(path, key=None):
    """Return a real vector and available rate metadata; never guess a MAT key."""
    path = Path(path)
    suffix = path.suffix.lower()
    metadata = {"path": str(path.resolve()), "key": None, "sample_rate": None}
    if suffix in (".wav", ".flac"):
        if key is not None:
            raise ValueError(f"A variable key is only supported for MAT files: {path}.")
        value, rate = sf.read(path, dtype="float64", always_2d=True)
        if value.shape[1] != 1:
            raise ValueError(f"Expected mono audio, got {value.shape[1]} channels: {path}.")
        value = value[:, 0]
        metadata["sample_rate"] = int(rate)
    elif suffix == ".npy":
        if key is not None:
            raise ValueError(f"A variable key is only supported for MAT files: {path}.")
        value = np.load(path, allow_pickle=False)
    elif suffix == ".mat":
        try:
            values = loadmat(path)
        except NotImplementedError as exc:
            raise ValueError(
                f"MAT v7.3/HDF5 is unsupported: {path}. Export the vector as MAT v7, NPY, or WAV."
            ) from exc
        metadata["sample_rate"] = _mat_rate(values, str(path))
        if key is None:
            candidates = []
            for name, candidate in values.items():
                if name.startswith("__") or name in _RATE_KEYS:
                    continue
                try:
                    _real_vector(candidate, name)
                except ValueError:
                    continue
                candidates.append(name)
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot select one MAT vector in {path}; candidates={candidates}. "
                    "Set the explicit variable key in the config."
                )
            key = candidates[0]
        if key not in values:
            raise ValueError(f"MAT variable {key!r} not found in {path}.")
        value = values[key]
        metadata["key"] = key
    else:
        raise ValueError(f"Unsupported vector file: {path}; use WAV, FLAC, MAT, or NPY.")
    value = _real_vector(value, str(path))
    metadata["frames"] = len(value)
    return value, metadata


def resolve_sample_rate(declared, observed, label):
    """Validate a declaration against file metadata, requiring at least one source."""
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, (int, np.integer)) or declared <= 0:
            raise ValueError(f"{label}: sample_rate must be a positive integer.")
        declared = int(declared)
    if declared is not None and observed is not None and declared != observed:
        raise ValueError(f"{label}: declared sample rate {declared} != file rate {observed}.")
    rate = declared if declared is not None else observed
    if rate is None:
        raise ValueError(f"{label}: sample rate is absent; set sample_rate explicitly, do not guess.")
    return int(rate)


def load_environments(cfg):
    """Return id -> arrays/provenance for independent PP/SP environments."""
    environments = {}
    for spec in cfg["environments"]:
        environment_id = str(spec["id"])
        if not environment_id or environment_id in environments:
            raise ValueError(f"Environment ids must be nonempty and unique: {environment_id!r}.")
        primary, p_info = read_vector(spec["primary_file"], spec.get("primary_key"))
        secondary, s_info = read_vector(spec["secondary_file"], spec.get("secondary_key"))
        expected_primary = spec.get("primary_length", cfg.get("primary_length"))
        expected_secondary = spec.get("secondary_length", cfg.get("secondary_length"))
        for name, value, expected in (("primary", primary, expected_primary),
                                      ("secondary", secondary, expected_secondary)):
            if expected is not None and len(value) != expected:
                raise ValueError(f"{environment_id}: expected {name} length {expected}, got {len(value)}.")
            if np.dot(value, value) <= 0:
                raise ValueError(f"{environment_id}: {name} path has zero energy.")
        declared = spec.get("sample_rate")
        if declared is None:
            declared = cfg.get("sample_rate")
        p_rate = resolve_sample_rate(declared, p_info["sample_rate"], f"{environment_id} PP")
        s_rate = resolve_sample_rate(declared, s_info["sample_rate"], f"{environment_id} SP")
        if p_rate != s_rate:
            raise ValueError(f"{environment_id}: PP and SP sample rates differ.")
        if cfg.get("sample_rate") is not None and p_rate != cfg["sample_rate"]:
            raise ValueError(f"{environment_id}: paths differ from dataset sample_rate.")
        p_info["sha256"] = sha256(spec["primary_file"])
        s_info["sha256"] = sha256(spec["secondary_file"])
        environments[environment_id] = {
            "id": environment_id, "primary": primary, "secondary": secondary,
            "sample_rate": p_rate, "provenance": {"primary": p_info, "secondary": s_info},
        }
    if not environments:
        raise ValueError("At least one environment is required.")
    return environments
