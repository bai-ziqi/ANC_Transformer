"""Configuration validation for clip targets and independent environment priors."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_sfanc_config(cfg, base):
    for key in ("manifest_file", "targets_dir"):
        cfg[key] = str((Path(base) / cfg[key]).resolve())
    for key in ("sample_rate", "filter_length", "primary_length", "secondary_length"):
        if not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be an explicit positive integer for SFANC.")
    environments = cfg.get("environments", [])
    ids = [item.get("id") for item in environments]
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("environments must have unique, nonempty string ids.")
    for item in environments:
        for key in ("primary_file", "secondary_file"):
            item[key] = str((Path(base) / item[key]).resolve())
        declared = item.get("sample_rate", cfg["sample_rate"])
        if declared is not None and declared != cfg["sample_rate"]:
            raise ValueError(f"Environment {item['id']} sample rate differs from the noise rate.")
    for split in ("train", "validation", "test"):
        values = cfg["protocol"].get(f"{split}_environments", [])
        if not values or len(set(values)) != len(values) or not set(values).issubset(ids):
            raise ValueError(f"protocol.{split}_environments must reference known environment ids.")
        if split not in cfg["sfanc"]["split_dirs"]:
            raise ValueError(f"sfanc.split_dirs is missing {split}.")
    cfg["sfanc"].setdefault("pairing", "round_robin")
    cfg["sfanc"].setdefault("waveform_key", None)
    cfg["sfanc"].setdefault("extensions", [".wav"])
    if cfg["sfanc"]["pairing"] not in ("round_robin", "all"):
        raise ValueError("sfanc.pairing must be round_robin or all.")
    if cfg["dataset"].get("score_mode") != "full_clip":
        raise ValueError("SFANC currently supports only the explicit offline full_clip protocol.")
    features = cfg["features"]
    obs = features.get("observation_seconds")
    if obs is not None and (not np.isfinite(obs) or obs <= 0):
        raise ValueError("observation_seconds must be positive or null (whole clip).")
    input_mode = features.get("input_mode", "log_spectrum")
    if input_mode not in ("log_spectrum", "waveform_conv"):
        raise ValueError("features.input_mode must be log_spectrum or waveform_conv.")
    if input_mode == "waveform_conv":
        waveform_defaults = {
            "conv_causal": True,
            "conv_kernel_size": 80,
            "conv_stride": 4,
            "conv_padding": 38,
            "conv_pool_kernel_size": 4,
            "conv_pool_stride": 4,
        }
        for key, value in waveform_defaults.items():
            features.setdefault(key, value)
        for key in ("conv_kernel_size", "conv_stride", "conv_padding",
                    "conv_pool_kernel_size", "conv_pool_stride"):
            if not isinstance(features[key], int) or features[key] <= 0:
                raise ValueError(f"features.{key} must be a positive integer.")
        if not isinstance(features["conv_causal"], bool):
            raise ValueError("features.conv_causal must be boolean.")
    for key in ("n_fft", "hop_length", "frequency_bins", "normalization_max_clips"):
        if not isinstance(features.get(key), int) or features[key] <= 0:
            raise ValueError(f"features.{key} must be a positive integer.")
    if features["hop_length"] > features["n_fft"] or features["frequency_bins"] > features["n_fft"] // 2 + 1:
        raise ValueError("Feature hop/bins exceed the feature FFT dimensions.")
    model = cfg["model"]
    if model.get("architecture") != "prior_adaptive" or model.get("prior_mode") != "sample":
        raise ValueError("SFANC requires prior_adaptive with explicit per-record prior_filter.")
    defaults = {"prior_tokenizer": "linear", "prior_normalize": False, "prior_hidden_dim": 256,
                "prior_frequency_embedding": True, "filter_output_mode": "direct", "filter_scale_mode": "unit"}
    for key, value in defaults.items():
        model.setdefault(key, value)
    for key in ("d_model", "nhead", "encoder_layers", "decoder_layers", "dim_feedforward",
                "prior_tokens", "prior_hidden_dim"):
        if not isinstance(model.get(key), int) or model[key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer.")
    pooling_mode = model.get("pooling_mode", "single_linear")
    pool_queries = model.get("pool_queries", 1)
    filter_head_hidden = model.get("filter_head_hidden", model["d_model"])
    return_pool_weights = model.get("return_pool_weights", False)
    for key, value in (("pool_queries", pool_queries),
                       ("filter_head_hidden", filter_head_hidden)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"model.{key} must be a positive integer.")
    if model["d_model"] % model["nhead"] or (model["d_model"] // model["nhead"]) % 2:
        raise ValueError("RoPE requires d_model divisible by nhead and an even head width.")
    if model["prior_tokenizer"] not in ("linear", "mlp", "fft_pool"):
        raise ValueError("Unknown prior_tokenizer.")
    if pooling_mode not in ("single_linear", "multi_query_attention"):
        raise ValueError("Unknown model.pooling_mode.")
    if model["filter_output_mode"] not in ("direct", "residual") or model["filter_scale_mode"] not in ("unit", "prior_rms"):
        raise ValueError("Unknown filter output or scale mode.")
    if (not isinstance(model["prior_normalize"], bool)
            or not isinstance(model["prior_frequency_embedding"], bool)
            or not isinstance(return_pool_weights, bool)):
        raise ValueError("Boolean model options must be boolean.")
    if not 0 <= model["dropout"] < 1 or model.get("rope_base", 0) <= 1:
        raise ValueError("Invalid dropout or RoPE base.")
    ridges = cfg["wiener"].get("ridge_ratios", [])
    if not ridges or any(not np.isfinite(value) or value < 0 for value in ridges):
        raise ValueError("ridge_ratios must be finite and nonnegative.")
    if cfg["wiener"].get("target_chunk_size", 0) <= 0:
        raise ValueError("target_chunk_size must be positive.")
    compare_toeplitz = cfg["wiener"].get("compare_toeplitz", True)
    if not isinstance(compare_toeplitz, bool):
        raise ValueError("wiener.compare_toeplitz must be boolean.")
    least_squares = cfg["wiener"].get("least_squares", {})
    if not isinstance(least_squares, dict):
        raise ValueError("wiener.least_squares must be a mapping.")
    positive = ("atol", "btol", "conlim", "normal_residual_tolerance")
    nonnegative = ("max_quantization_relative_increase",
                   "control_comparison_relative_tolerance")
    for key in positive:
        value = least_squares.get(key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"wiener.least_squares.{key} must be finite and positive.")
    for key in nonnegative:
        value = least_squares.get(key)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"wiener.least_squares.{key} must be finite and nonnegative.")
    maxiter = least_squares.get("maxiter", 2000)
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0:
        raise ValueError("wiener.least_squares.maxiter must be a positive integer.")
    prior = cfg["prior"]
    if not 0 < prior["low_hz"] < prior["high_hz"] < cfg["sample_rate"] / 2:
        raise ValueError("Prior band must lie strictly below the Nyquist frequency.")
    for key in ("fit_seconds", "validation_seconds", "warmup_seconds"):
        if not np.isfinite(prior[key]) or prior[key] <= 0:
            raise ValueError(f"prior.{key} must be positive.")
    if min(round(prior[key] * cfg["sample_rate"]) for key in ("fit_seconds", "validation_seconds")) <= cfg["filter_length"]:
        raise ValueError("Prior fit/validation signals must exceed the control FIR length.")
    training = cfg["training"]
    for key in ("epochs", "batch_size"):
        if not isinstance(training.get(key), int) or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    for key in ("learning_rate", "gradient_clip"):
        if not np.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be positive.")
    mode = training.get("loss_mode")
    if mode not in ("filter", "acoustic", "combined"):
        raise ValueError("Unknown training loss_mode.")
    for key in ("filter_loss_weight", "acoustic_loss_weight", "weight_decay"):
        if not np.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative.")
    if ((mode == "filter" and training["filter_loss_weight"] == 0) or
            (mode == "acoustic" and training["acoustic_loss_weight"] == 0) or
            (mode == "combined" and training["filter_loss_weight"] + training["acoustic_loss_weight"] == 0)):
        raise ValueError("At least one active training loss must have positive weight.")
    if not isinstance(training.get("num_workers"), int) or training["num_workers"] < 0:
        raise ValueError("training.num_workers must be a nonnegative integer.")
    return cfg
