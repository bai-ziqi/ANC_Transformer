"""Clip-label/prior separation and real 16-kHz, 512/256/1024-tap integration."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from scipy.io import savemat
from scipy.signal import lfilter

from anc.anc_simulator import simulate_numpy
from data.environments import load_environments, read_vector
from data.sfanc import SfancDataset, inspect_sfanc, load_training_bundle
from experiments.common import evaluate, load_model, losses, save_evaluation
from experiments.train import train
from utils import dump_json, load_config
from wiener.build_priors import build_priors
from wiener.build_targets import build_targets
from wiener.calculate_wiener import fit_wiener_array
from wiener.diagnose_labels import diagnose_labels


@pytest.fixture
def sfanc_config(tmp_path):
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1] / "config.sfanc.yaml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key, name in (("data_root", "noise"), ("manifest_file", "results/manifest.json"),
                      ("targets_dir", "targets"), ("bank_file", "priors.pt"),
                      ("output_dir", "results"), ("checkpoint_file", "model.pt")):
        cfg[key] = str(tmp_path / name)
    primary, secondary = np.zeros(512), np.zeros(256)
    secondary[:3] = [0.8, -0.08, 0.02]
    expected_filter = np.zeros(20)
    expected_filter[[5, 6, 9]] = [0.5, -0.1, 0.03]
    product = np.convolve(secondary, expected_filter)
    primary[:len(product)] = product
    p_file, s_file = tmp_path / "Primary_path.mat", tmp_path / "Secondary_path.mat"
    savemat(p_file, {"P": primary[:, None], "fs": 16000})
    savemat(s_file, {"S": secondary[None], "Fs": 16000})
    cfg["environments"][0].update(primary_file=str(p_file), secondary_file=str(s_file))
    counts = {"train": 2, "validation": 1, "test": 1}
    cfg["sfanc"]["expected_counts"] = counts
    rng = np.random.default_rng(54)
    for split, count in counts.items():
        directory = Path(cfg["data_root"]) / cfg["sfanc"]["split_dirs"][split]
        directory.mkdir(parents=True)
        for index in range(count):
            value = lfilter([1.0], [1.0, -0.3], rng.normal(0, 0.05, 4096))
            sf.write(directory / f"noise_{index}.wav", value, 16000, subtype="FLOAT")
    cfg["wiener"].update(ridge_ratios=[0.0, 0.0001, 0.1], target_chunk_size=2)
    cfg["prior"].update(fit_seconds=0.3, validation_seconds=0.2, warmup_seconds=0.1)
    cfg["features"].update(n_fft=256, hop_length=64, frequency_bins=33, normalization_max_clips=2)
    cfg["model"].update(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                        dim_feedforward=16, prior_tokens=4, dropout=0.0)
    cfg["training"].update(epochs=1, batch_size=2, device="cpu")
    cfg["metrics"].update(n_fft=512, hop_length=128)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_config(path)


def test_candidate_selection_is_actual_same_clip_mse():
    rng = np.random.default_rng(4)
    x = lfilter([1.0], [1.0, -0.8], rng.normal(size=3000))
    p, s = np.array([0.0, 0.0, 0.5, -0.15]), np.array([0.7, -0.2])
    coefficients, selected, rows = fit_wiener_array(x, p, s, 32, [0.0, 0.0001, 0.01, 1.0])
    disturbance = lfilter(p, [1], x)
    mse = np.mean(simulate_numpy(x, disturbance, coefficients, s)["residual"] ** 2)
    assert selected["residual_mse"] == pytest.approx(mse, rel=1e-12)
    assert mse == pytest.approx(min(row["residual_mse"] for row in rows), rel=1e-12)
    assert selected["absolute_ridge"] == selected["ridge_ratio"] * selected["filtered_reference_power"]


def test_label_diagnostics_compare_cached_prior_and_exact_window(sfanc_config):
    build_priors(sfanc_config)
    build_targets(sfanc_config)
    report, filters = diagnose_labels(
        sfanc_config, split="validation", indices=[0], count=None,
        ridge_ratios=[0.0, 0.1], maxiter=1000,
    )
    assert report["scope"].startswith("read-only")
    assert report["summary"]["samples"] == 1
    sample = report["samples"][0]
    assert abs(sample["cached_label"]["rescoring_relative_difference"]) < 1e-8
    assert len(sample["exact_finite_window_candidates"]) == 2
    assert filters["prior_filters"].shape == (1, sfanc_config["filter_length"])
    assert filters["exact_ls_filters"].shape == (1, 2, sfanc_config["filter_length"])


def test_default_config_uses_sfanc_dimensions_without_scene_classes():
    cfg = load_config()
    assert cfg["protocol"]["mode"] == "sfanc"
    assert cfg["sample_rate"] == 16000
    assert cfg["primary_length"] == 512
    assert cfg["secondary_length"] == 256
    assert cfg["filter_length"] == 1024
    assert "scene_names" not in cfg
    assert "path_id" not in cfg
    assert Path(cfg["bank_file"]).is_absolute()


def test_mat_keys_lengths_and_sample_rate(sfanc_config, tmp_path):
    env = load_environments(sfanc_config)["path_1"]
    assert env["primary"].shape == (512,)
    assert env["secondary"].shape == (256,)
    assert env["secondary"][0] == 0.8
    ambiguous = tmp_path / "ambiguous.mat"
    savemat(ambiguous, {"first": np.ones(512), "second": np.ones(512)})
    with pytest.raises(ValueError, match="explicit variable key"):
        read_vector(ambiguous)
    vector, _ = read_vector(ambiguous, "second")
    assert vector.shape == (512,)
    savemat(sfanc_config["environments"][0]["secondary_file"], {"S": np.ones(256), "Fs": 48000})
    with pytest.raises(ValueError, match="sample rate"):
        load_environments(sfanc_config)


def test_filter_supervision_uses_target_not_prior(sfanc_config):
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    result = {"filter": prediction, "prior_filter": torch.tensor([[9.0, 9.0]])}
    batch = {"target_filter": prediction.detach().clone()}
    outcome = losses(result, sfanc_config, batch)
    assert outcome["loss"].item() == 0
    assert "acoustic_mse" not in outcome
    with pytest.raises(ValueError, match="target_filter"):
        losses(result, sfanc_config, {})


def test_prior_is_independent_of_dataset_and_repeatable(sfanc_config):
    cfg = copy.deepcopy(sfanc_config)
    cfg["data_root"] = str(Path(cfg["data_root"]) / "does_not_exist")
    first, second = build_priors(cfg), build_priors(cfg)
    torch.testing.assert_close(first["prior_filters"], second["prior_filters"], rtol=0, atol=0)
    assert first["prior_filters"].shape == (1, 1024)
    assert first["primary_paths"].shape == (1, 512)
    assert first["secondary_paths"].shape == (1, 256)


def test_sfanc_end_to_end_and_resume(sfanc_config):
    priors = build_priors(sfanc_config)
    index = build_targets(sfanc_config)
    assert index["complete"] and index["total_records"] == 4
    assert index["selection_metric"] == "minimum_full_window_causal_residual_mse"
    assert index["solver"] == "pure_full_window_lsmr"
    for chunk in index["chunks"]:
        diagnostics = json.loads((Path(sfanc_config["targets_dir"]) / chunk["diagnostics_file"]).read_text())
        for record in diagnostics:
            ls = record["least_squares"]
            assert ls["acceptance"] == "passed"
            assert ls["prior_anchor_used"] is False
            assert ls["absolute_ridge"] == 0.0
            assert ls["normal_equation_backward_error"] <= ls["normal_residual_tolerance"]
            comparison = record["environment_prior_comparison"]
            assert comparison["pure_ls_minus_control_relative"] <= \
                sfanc_config["wiener"]["least_squares"]["control_comparison_relative_tolerance"]
            if record["toeplitz_comparison"] is not None:
                assert record["toeplitz_comparison"]["pure_ls_minus_control_relative"] <= \
                    sfanc_config["wiener"]["least_squares"]["control_comparison_relative_tolerance"]
    assert build_targets(sfanc_config, resume=True) == index
    bundle = load_training_bundle(sfanc_config)
    train_data = SfancDataset(sfanc_config, bundle, "train")
    item = train_data[0]
    assert item["observation"].shape == item["reference"].shape == (4096,)
    assert item["target_filter"].shape == item["prior_filter"].shape == (1024,)
    torch.testing.assert_close(item["prior_filter"], priors["prior_filters"][0].float())
    assert item["secondary_path"].shape == (256,)
    assert item["context"] == 0
    checkpoint = train(sfanc_config)
    assert checkpoint["format_version"] == 5
    model, bundle, loaded = load_model(sfanc_config)
    methods = ("zero_filter", "environment_prior", "sample_wiener", "transformer")
    rows = evaluate(sfanc_config, model, bundle, "validation", methods)
    assert {row["method"] for row in rows} == set(methods)
    zero = next(row for row in rows if row["method"] == "zero_filter")
    assert zero["residual_mse"] == pytest.approx(zero["disturbance_mse"], rel=1e-12)
    assert zero["broadband_nr_db"] == pytest.approx(0.0)
    target = next(row for row in rows if row["method"] == "sample_wiener")
    assert target["relative_filter_mse"] == 0.0
    summary = save_evaluation(sfanc_config, rows, "validation", "comparison", loaded)
    assert len(summary) == 8
    with pytest.raises(ValueError, match="new targets_dir"):
        build_targets(sfanc_config, limit=1, resume=True)


def test_partial_cache_is_explicit_and_opt_in(sfanc_config):
    build_priors(sfanc_config)
    index = build_targets(sfanc_config, limit=1)
    assert index["partial"] and index["total_records"] == 3
    with pytest.raises(ValueError, match="partial|Partial"):
        load_training_bundle(sfanc_config)
    sfanc_config["training"]["allow_partial"] = True
    bundle = load_training_bundle(sfanc_config)
    assert len(SfancDataset(sfanc_config, bundle, "train")) == 1


def test_legacy_toeplitz_target_cache_is_selected_explicitly(sfanc_config):
    build_priors(sfanc_config)
    build_targets(sfanc_config)

    sfanc_config["target_cache_mode"] = "toeplitz"
    index_path = Path(sfanc_config["targets_dir"]) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["format_version"] = 1
    index["selection_metric"] = "minimum_causal_residual_mse"
    index["solver"] = "biased_stationary_toeplitz"
    dump_json(index_path, index)

    bundle = load_training_bundle(sfanc_config)
    assert bundle["targets_index"]["solver"] == "biased_stationary_toeplitz"

    sfanc_config["target_cache_mode"] = "finite_window_ls"
    with pytest.raises(ValueError, match="target_cache_mode=finite_window_ls"):
        load_training_bundle(sfanc_config)


def test_epoch_zero_is_selected_when_training_does_not_improve(sfanc_config, monkeypatch):
    sfanc_config["model"]["filter_output_mode"] = "residual"
    build_priors(sfanc_config)
    build_targets(sfanc_config)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *args, **kwargs: None)
    selected = train(sfanc_config)
    history_file = Path(sfanc_config["output_dir"]) / "training_history.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [0, 1]
    assert history[0]["baseline"] is True
    assert history[0]["optimizer_steps"] == 0
    assert selected["epoch"] == 0
    assert selected["validation_loss"] == pytest.approx(history[0]["validation_loss"])


def test_cache_resume_checks_corruption(sfanc_config):
    build_priors(sfanc_config)
    build_targets(sfanc_config)
    index_path = Path(sfanc_config["targets_dir"]) / "index.json"
    index = json.loads(index_path.read_text())
    index["chunks"][0]["sha256"] = "invalid"
    dump_json(index_path, index)
    with pytest.raises(ValueError, match="corrupt|changed"):
        build_targets(sfanc_config, resume=True)


def test_runtime_cache_and_features_do_not_invalidate_labels(sfanc_config):
    build_priors(sfanc_config)
    index = build_targets(sfanc_config)
    sfanc_config["sfanc"]["target_cache_chunks"] = 1
    sfanc_config["features"]["normalization_max_clips"] = 1
    assert build_targets(sfanc_config, resume=True) == index
    bundle = load_training_bundle(sfanc_config)
    assert len(bundle["normalizer"]["sample_ids"]) == 1


def test_changed_source_is_rejected(sfanc_config):
    manifest = inspect_sfanc(sfanc_config)
    record = manifest["records"][0]
    sf.write(record["path"], np.zeros(4096), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="sample rate"):
        inspect_sfanc(sfanc_config)
