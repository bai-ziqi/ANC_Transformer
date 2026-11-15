"""Check model tensor shapes with synthetic inputs, without loading server data."""
from __future__ import annotations

import argparse
import json

import torch

from models.anc_transformer import ANCTransformer
from utils import device_for, load_config, seed_everything


def inspect_model(cfg, samples=None, batch_size=2, device="cpu"):
    """Return dimensions only; dummy paths and normalization are not calibration."""
    if cfg.get("protocol", {}).get("mode") != "sfanc":
        raise ValueError("This model-entry check requires an SFANC configuration.")
    if samples is None:
        seconds = cfg["features"].get("observation_seconds")
        samples = round((1.0 if seconds is None else seconds) * cfg["sample_rate"])
    if not isinstance(samples, int) or samples < cfg["features"]["n_fft"]:
        raise ValueError("samples must cover at least the feature FFT.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    seed_everything(cfg["seed"])
    bins, taps = cfg["features"]["frequency_bins"], cfg["filter_length"]
    primary = torch.zeros(1, cfg["primary_length"], dtype=torch.float64)
    secondary = torch.zeros(1, cfg["secondary_length"], dtype=torch.float64)
    primary[:, 0], secondary[:, 0] = 1.0, 1.0
    bundle = {
        "format_version": 3, "sample_rate": cfg["sample_rate"], "filter_length": taps,
        "primary_length": cfg["primary_length"], "secondary_length": cfg["secondary_length"],
        "environment_ids": ["shape_check_only"], "primary_paths": primary,
        "secondary_paths": secondary, "prior_filters": torch.zeros(1, taps),
        "normalizer": {"mean": torch.zeros(bins), "std": torch.ones(bins)},
    }
    model = ANCTransformer(cfg, bundle).to(device_for(device)).eval()
    observation = torch.randn(batch_size, samples)
    prior = torch.randn(batch_size, taps) * 0.01
    shapes = {}

    def capture(name):
        def hook(module, inputs, output):
            shapes[name] = list(output.shape)
        return hook

    hooks = [model.encoder.register_forward_hook(capture("encoder_tokens_shape")),
             model.decoder.prior_tokenizer.register_forward_hook(capture("prior_tokens_shape"))]
    try:
        with torch.no_grad():
            output = model(observation, prior_filter=prior)["filter"]
    finally:
        for hook in hooks:
            hook.remove()
    if output.shape != (batch_size, taps) or not torch.isfinite(output).all():
        raise RuntimeError("Model entry produced an invalid FIR output.")
    return {
        "data_kind": "synthetic shape check only; not trained or acoustically calibrated",
        "sample_rate": cfg["sample_rate"], "primary_length": cfg["primary_length"],
        "secondary_length": cfg["secondary_length"], "filter_length": taps,
        "primary_path_shape": list(primary.shape), "secondary_path_shape": list(secondary.shape),
        "observation_shape": list(observation.shape), "prior_shape": list(prior.shape),
        **shapes, "output_shape": list(output.shape), "target_shape": [batch_size, taps],
        "device": str(next(model.parameters()).device),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="config.sfanc.yaml")
    parser.add_argument("--samples", type=int, help="Observation sample count; default is one second for full-clip mode.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = inspect_model(load_config(args.config), args.samples, args.batch_size, args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
