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


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.sfanc.yaml")


def load_config(path=DEFAULT_CONFIG):
    from data.sfanc_config import validate_sfanc_config

    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("protocol", {}).get("mode") != "sfanc":
        raise ValueError("Use an SFANC configuration with protocol.mode=sfanc.")
    for key in ("data_root", "output_dir", "bank_file", "checkpoint_file"):
        cfg[key] = str((source.parent / cfg[key]).resolve())
    cfg["config_file"] = str(source)
    return validate_sfanc_config(cfg, source.parent)


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
    """Settings defining environment priors and sample targets."""
    keys = ("sample_rate", "filter_length", "primary_length", "secondary_length",
            "data_root", "sfanc", "environments", "protocol", "wiener", "prior")
    signature = copy.deepcopy({key: cfg[key] for key in keys})
    # Runtime cache limits and encoder features do not change the Wiener FIRs.
    signature["sfanc"].pop("target_cache_chunks", None)
    return signature
