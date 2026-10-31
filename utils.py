"""Configuration, provenance, and reproducibility shared by all commands."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.yaml")


def load_config(path=DEFAULT_CONFIG):
    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key in ("data_root", "output_dir", "bank_file", "checkpoint_file"):
        cfg[key] = str((source.parent / cfg[key]).resolve())
    cfg["config_file"] = str(source)
    if cfg["sample_rate"] != 48000 or cfg["filter_length"] != 1967:
        raise ValueError("V1 retains 48 kHz audio and exactly 1967 taps.")
    names = cfg["scene_names"]
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError("Exactly eight distinct scenes are required.")
    if not isinstance(cfg["path_id"], int) or not 1 <= cfg["path_id"] <= 10:
        raise ValueError("path_id must be an integer in 1..10.")
    fractions = cfg["split"]["fractions"]
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("split.fractions must be three positive values summing to one.")
    for key in ("skip_seconds", "tail_margin_seconds", "gap_seconds", "calibration_gap_seconds"):
        if cfg["split"][key] < 0:
            raise ValueError(f"split.{key} cannot be negative.")
    if cfg["split"]["calibration_seconds"] <= 0:
        raise ValueError("calibration_seconds must be positive.")
    for key in ("observation_seconds", "hop_length", "n_fft", "frequency_bins", "normalization_windows_per_scene"):
        if cfg["features"][key] <= 0:
            raise ValueError(f"features.{key} must be positive.")
    obs = round(cfg["features"]["observation_seconds"] * cfg["sample_rate"])
    if obs < max(cfg["features"]["n_fft"], 2 * (cfg["filter_length"] - 1)):
        raise ValueError("Observation must cover feature FFT and cascade history.")
    if cfg["dataset"]["control_seconds"] <= 0:
        raise ValueError("control_seconds must be positive.")
    for split in ("train", "validation", "test"):
        if cfg["dataset"][f"{split}_windows_per_scene"] <= 0:
            raise ValueError("Window counts must be positive.")
    if min(cfg["model"][k] for k in ("d_model", "nhead", "encoder_layers", "decoder_layers", "dim_feedforward")) <= 0:
        raise ValueError("Model dimensions/layer counts must be positive.")
    if cfg["model"]["d_model"] % cfg["model"]["nhead"]:
        raise ValueError("d_model must be divisible by nhead.")
    if not 0 <= cfg["model"]["dropout"] < 1:
        raise ValueError("dropout must be in [0,1).")
    if cfg["training"]["epochs"] <= 0 or cfg["training"]["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if cfg["training"]["learning_rate"] <= 0 or cfg["training"]["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive.")
    if cfg["training"]["filter_loss_weight"] < 0 or cfg["training"]["num_workers"] < 0:
        raise ValueError("filter loss weight and num_workers cannot be negative.")
    if not 0 < cfg["features"]["hop_length"] <= cfg["features"]["n_fft"]:
        raise ValueError("Feature hop must be in 1..n_fft.")
    if not 0 < cfg["features"]["frequency_bins"] <= cfg["features"]["n_fft"] // 2 + 1:
        raise ValueError("frequency_bins may pool but must not upsample the FFT.")
    if cfg["wiener"]["fit_seconds_per_scene"] <= 0 or cfg["wiener"]["fit_segments"] <= 0:
        raise ValueError("Wiener fit duration and segment count must be positive.")
    ridges = cfg["wiener"]["ridge_ratios"]
    if not ridges or any(not np.isfinite(v) or v < 0 for v in ridges):
        raise ValueError("ridge_ratios must be finite and nonnegative.")
    return cfg


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path):
    # Artifacts contain only tensors and JSON-like primitives; never unpickle modules.
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def bank_signature_cfg(cfg):
    """Settings defining the bank; changes to training epochs do not invalidate it."""
    return copy.deepcopy({k: cfg[k] for k in ("path_id", "sample_rate", "filter_length", "scene_names", "split", "wiener", "features", "metrics")})
