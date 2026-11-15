"""Model contracts for separate PP512, SP256, and 1024-tap control filters."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.common import losses
from models.anc_transformer import ANCTransformer
from utils import load_config


@pytest.fixture
def model_inputs():
    torch.set_num_threads(1)
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.sfanc.yaml")
    primary, secondary = torch.zeros(1, 512, dtype=torch.float64), torch.zeros(1, 256, dtype=torch.float64)
    primary[0, 8], secondary[0, 2] = 0.5, 0.8
    prior = torch.randn(1, 1024, dtype=torch.float64) * 0.01
    bundle = {
        "format_version": 3,
        "sample_rate": 16000,
        "primary_length": 512,
        "secondary_length": 256,
        "filter_length": 1024,
        "environment_ids": ["path_1"],
        "prior_filters": prior,
        "primary_paths": primary,
        "secondary_paths": secondary,
        "normalizer": {"mean": torch.zeros(129), "std": torch.ones(129)},
    }
    return cfg, bundle


def test_default_sfanc_dimensions_and_gradient_flow(model_inputs):
    cfg, bundle = model_inputs
    model = ANCTransformer(cfg, bundle).eval()
    observation = torch.randn(2, 16000)
    prior = bundle["prior_filters"].float().expand(2, -1).clone().requires_grad_()
    captured = []
    hook = model.decoder.prior_tokenizer.register_forward_hook(
        lambda module, inputs, output: captured.append(tuple(output.shape))
    )
    prediction = model(observation, prior_filter=prior)
    hook.remove()
    assert prediction["filter"].shape == (2, 1024)
    assert captured == [(2, 32, 64)]
    assert model.decoder.prior_tokenizer.projection.in_features == 1024
    prediction["filter"].square().mean().backward()
    assert prior.grad is not None and torch.isfinite(prior.grad).all()
    assert model.filter_head.output[-1].weight.grad is not None


def test_short_observation_and_shared_prior_are_supported(model_inputs):
    cfg, bundle = model_inputs
    model = ANCTransformer(cfg, bundle).eval()
    observation = torch.randn(2, cfg["features"]["n_fft"])
    prediction = model(observation, prior_filter=bundle["prior_filters"][0])
    assert prediction["filter"].shape == (2, 1024)
    with pytest.raises(ValueError):
        model(observation[:, :-1], prior_filter=bundle["prior_filters"][0])


@pytest.mark.parametrize("taps", [256, 512])
def test_path_length_cannot_be_used_as_control_prior(model_inputs, taps):
    cfg, bundle = model_inputs
    model = ANCTransformer(cfg, bundle)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1024), prior_filter=torch.zeros(2, taps))


@pytest.mark.parametrize("kind", ["unbatched", "integer", "complex"])
def test_observation_requires_batched_real_floating_samples(model_inputs, kind):
    cfg, bundle = model_inputs
    model = ANCTransformer(cfg, bundle)
    if kind == "unbatched":
        observation = torch.zeros(16000)
    else:
        dtype = torch.int64 if kind == "integer" else torch.complex64
        observation = torch.zeros(2, 16000, dtype=dtype)
    with pytest.raises((ValueError, TypeError)):
        model(observation, prior_filter=bundle["prior_filters"][0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_loaded_float64_waveforms_use_model_parameter_dtype(model_inputs, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test environment.")
    cfg, bundle = model_inputs
    model = ANCTransformer(cfg, bundle).to(device).eval()
    prediction = model(torch.randn(2, 1024, dtype=torch.float64),
                       prior_filter=bundle["prior_filters"][0])
    assert prediction["filter"].dtype == torch.float32
    assert prediction["prior_filter"].dtype == torch.float32
    assert prediction["filter"].device.type == device
    assert prediction["prior_filter"].device.type == device


@pytest.mark.parametrize("invalid", ["broadcast_mean", "zero_std", "nonfinite_std"])
def test_normalizer_must_match_feature_bins_without_broadcast(model_inputs, invalid):
    cfg, original = model_inputs
    bundle = copy.deepcopy(original)
    if invalid == "broadcast_mean":
        bundle["normalizer"]["mean"] = torch.zeros(1)
    elif invalid == "zero_std":
        bundle["normalizer"]["std"][0] = 0
    else:
        bundle["normalizer"]["std"][0] = float("nan")
    with pytest.raises(ValueError):
        ANCTransformer(cfg, bundle)


def test_path_length_is_not_control_filter_length(model_inputs):
    cfg, bundle = model_inputs
    bundle["secondary_paths"] = torch.zeros(1, 1024, dtype=torch.float64)
    with pytest.raises(ValueError):
        ANCTransformer(cfg, bundle)


def test_filter_target_cannot_silently_broadcast(model_inputs):
    cfg, _ = model_inputs
    prediction = {"filter": torch.zeros(2, 1024, requires_grad=True)}
    with pytest.raises(ValueError):
        losses(prediction, cfg, {"target_filter": torch.ones(2, 1)})


def test_model_inspection_needs_no_server_data(model_inputs):
    from models.inspect_model import inspect_model

    cfg, _ = model_inputs
    report = inspect_model(cfg, samples=16000, batch_size=2, device="cpu")
    assert report["sample_rate"] == 16000
    assert report["primary_length"] == 512
    assert report["secondary_length"] == 256
    assert report["filter_length"] == 1024
    assert tuple(report["observation_shape"]) == (2, 16000)
    assert tuple(report["prior_shape"]) == (2, 1024)
    assert tuple(report["prior_tokens_shape"]) == (2, 32, 64)
    assert tuple(report["output_shape"]) == (2, 1024)
