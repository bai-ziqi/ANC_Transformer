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


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.cross_path.yaml")


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
    architecture = cfg["model"].get("architecture")
    if architecture not in ("prior_adaptive", "legacy_bank_fusion"):
        raise ValueError("model.architecture must be prior_adaptive or legacy_bank_fusion.")
    if architecture == "prior_adaptive":
        if cfg["model"].get("prior_mode") != "sample":
            raise ValueError("model.prior_mode must be sample; priors are never pooled.")
        if cfg["model"].get("prior_tokens", 0) <= 0:
            raise ValueError("model.prior_tokens must be positive.")
        if cfg["model"].get("rope_base", 0) <= 1:
            raise ValueError("model.rope_base must be greater than one.")
        head_width = cfg["model"]["d_model"] // cfg["model"]["nhead"]
        if head_width % 2:
            raise ValueError("RoPE requires an even width per attention head.")
        cfg["model"].setdefault("filter_output_mode", "direct")
        if cfg["model"]["filter_output_mode"] not in ("direct", "residual"):
            raise ValueError("model.filter_output_mode must be direct or residual.")
    if not 0 <= cfg["model"]["dropout"] < 1:
        raise ValueError("dropout must be in [0,1).")
    if cfg["training"]["epochs"] <= 0 or cfg["training"]["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if cfg["training"]["learning_rate"] <= 0 or cfg["training"]["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive.")
    if cfg["training"].get("loss_mode") not in ("acoustic", "filter", "combined"):
        raise ValueError("training.loss_mode must be acoustic, filter, or combined.")
    if cfg["training"].get("filter_loss_weight", -1) < 0:
        raise ValueError("training.filter_loss_weight cannot be negative.")
    if cfg["training"].get("acoustic_loss_weight", -1) < 0:
        raise ValueError("training.acoustic_loss_weight cannot be negative.")
    if cfg["training"]["num_workers"] < 0:
        raise ValueError("training.num_workers cannot be negative.")
    if not 0 < cfg["features"]["hop_length"] <= cfg["features"]["n_fft"]:
        raise ValueError("Feature hop must be in 1..n_fft.")
    if not 0 < cfg["features"]["frequency_bins"] <= cfg["features"]["n_fft"] // 2 + 1:
        raise ValueError("frequency_bins may pool but must not upsample the FFT.")
    if cfg["wiener"]["fit_seconds_per_scene"] <= 0 or cfg["wiener"]["fit_segments"] <= 0:
        raise ValueError("Wiener fit duration and segment count must be positive.")
    ridges = cfg["wiener"]["ridge_ratios"]
    if not ridges or any(not np.isfinite(v) or v < 0 for v in ridges):
        raise ValueError("ridge_ratios must be finite and nonnegative.")
    protocol = cfg.get("protocol")
    if protocol is not None:
        if protocol.get("mode") != "cross_path":
            raise ValueError("protocol.mode must be cross_path when protocol is set.")
        groups = {}
        for split in ("train", "validation", "test"):
            values = protocol.get(f"{split}_paths")
            if not values or any(not isinstance(value, int) or not 1 <= value <= 10 for value in values):
                raise ValueError(f"protocol.{split}_paths must contain path ids in 1..10.")
            if len(values) != len(set(values)):
                raise ValueError(f"protocol.{split}_paths cannot contain duplicates.")
            groups[split] = set(values)
        if (groups["train"] & groups["validation"] or
                groups["train"] & groups["test"] or
                groups["validation"] & groups["test"]):
            raise ValueError("protocol path splits must be disjoint.")
        if len(groups["train"]) < 2:
            raise ValueError("Cross-path training requires at least two training paths.")
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
    signature = copy.deepcopy({k: cfg[k] for k in ("path_id", "sample_rate", "filter_length", "scene_names", "split", "wiener", "features", "metrics")})
    if "protocol" in cfg:
        signature["protocol"] = copy.deepcopy(cfg["protocol"])
    return signature
