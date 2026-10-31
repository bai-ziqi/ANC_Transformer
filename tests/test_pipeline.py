"""Scientific invariants and a small end-to-end, 48-kHz/1967-tap experiment."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from scipy.signal import lfilter

from anc.anc_simulator import simulate, simulate_numpy
from anc.metrics import power_spectrum, score_signals
from data.dataset import ANCWindows
from data.inspect_data import inspect, read_mono
from experiments.common import evaluate, load_model, verified_bundle
from experiments.train import train
from models.anc_transformer import ANCTransformer
from utils import DEFAULT_CONFIG, load_config
from wiener.build_bank import build_bank
from wiener.calculate_wiener import energy_diagnostics, solve_candidates, statistic_vectors


@pytest.fixture
def tiny_config(tmp_path):
    torch.set_num_threads(1)
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["data_root"] = str(tmp_path / "audio")
    cfg["output_dir"] = str(tmp_path / "results")
    cfg["bank_file"] = str(tmp_path / "bank.pt")
    cfg["checkpoint_file"] = str(tmp_path / "best.pt")
    cfg["split"].update(skip_seconds=0.1, tail_margin_seconds=0.1, gap_seconds=0.05,
                         calibration_seconds=0.3, calibration_gap_seconds=0.05)
    cfg["wiener"].update(fit_seconds_per_scene=0.3, fit_segments=1, ridge_ratios=[0.001, 0.01])
    cfg["features"].update(observation_seconds=0.1, n_fft=512, hop_length=256,
                            frequency_bins=33, normalization_windows_per_scene=1)
    cfg["dataset"].update(control_seconds=0.1, train_windows_per_scene=1,
                           validation_windows_per_scene=1, test_windows_per_scene=1)
    cfg["model"].update(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                         dim_feedforward=16, dropout=0.0)
    cfg["training"].update(epochs=1, batch_size=8, device="cpu")
    cfg["metrics"].update(n_fft=8192, hop_length=2048)
    root = Path(cfg["data_root"])
    (root / "NOISE").mkdir(parents=True)
    (root / "EXPECTED_NOISE").mkdir()
    rng = np.random.default_rng(7)
    secondary = np.zeros((1967, 10))
    secondary[0] = 1.0
    secondary[20] = 0.05
    np.save(root / "sh.npy", secondary)
    for i, name in enumerate(cfg["scene_names"]):
        x = rng.normal(0, 0.05, 6 * 48000)
        x = lfilter([1], [1, -0.08 * i], x)
        d = lfilter([0.4 + 0.02 * i, 0.1], [1], x)
        sf.write(root / "NOISE" / f"{name}.wav", x, 48000, subtype="FLOAT")
        sf.write(root / "EXPECTED_NOISE" / f"{name}_scene_01.wav", d, 48000, subtype="FLOAT")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return load_config(path)


def test_wiener_lag_sign_and_energy():
    rng = np.random.default_rng(1)
    z = rng.normal(size=100000)
    expected = np.array([0.7, -0.2, 0.08])
    d = lfilter(expected, [1], z)
    auto, cross = statistic_vectors(z, d, 3)
    actual = solve_candidates(auto, cross, [0.0])[0.0]
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    energy = energy_diagnostics(np.array([1.0, 0.0, 0.0]))
    assert energy["taps_for_95_percent"] == energy["taps_for_99_percent"] == 1


def test_causal_simulation_does_not_invent_past_actuator_state():
    x = np.arange(12, dtype=np.float64) / 10
    d = np.ones_like(x)
    w = np.array([0.4, -0.1, 0.2])
    s = np.array([1.0, 0.3])
    context = 2
    y = lfilter(w, [1], x)
    y[:context] = 0
    expected = d - lfilter(s, [1], y)
    numpy_result = simulate_numpy(x, d, w, s, context)
    tensor_result = simulate(torch.tensor(x[None]), torch.tensor(d[None]),
                             torch.tensor(w[None]), torch.tensor(s), context)
    np.testing.assert_allclose(numpy_result["residual"], expected[context:], atol=1e-14)
    np.testing.assert_allclose(tensor_result["residual"][0].numpy(), expected[context:], atol=1e-14)
    # At the decision boundary only y[t] passes through s[0]; y[t-1] is zero.
    assert numpy_result["anti_noise"][0] == pytest.approx(y[context])


def test_metric_known_gain_and_band_rebound():
    x = np.random.default_rng(3).normal(size=24000)
    metrics = score_signals(x, x * 0.5, x * 0)
    assert metrics["primary_nr_db"] == pytest.approx(20 * np.log10(2), abs=1e-9)
    assert metrics["broadband_nr_db"] == pytest.approx(20 * np.log10(2), abs=1e-9)
    assert metrics["maximum_rebound_db"] == 0
    amplified = score_signals(x, x * 2, x * 0)
    assert amplified["maximum_rebound_db"] == pytest.approx(20 * np.log10(2), abs=1e-9)


def test_spectrum_matches_original_torch_scoring_convention():
    x = np.random.default_rng(17).normal(size=24000)
    waveform = torch.tensor(x, dtype=torch.float64)
    waveform = waveform - waveform.mean()
    n_fft, hop = 8192, 2048
    padded = max(len(x), n_fft)
    padded += (-(padded - n_fft)) % hop
    waveform = torch.nn.functional.pad(waveform, (0, padded - len(x)))
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    spectrum = torch.stft(waveform, n_fft, hop, window=window, center=False, return_complex=True)
    expected = spectrum.abs().square().mean(-1) / window.square().sum()
    expected[1:-1] *= 2
    np.testing.assert_allclose(power_spectrum(x, n_fft, hop), expected.numpy(), rtol=1e-12, atol=1e-12)


def test_splits_reads_and_bank_are_train_only(tiny_config, monkeypatch):
    cfg = tiny_config
    manifest = inspect(cfg)
    allowed = {}
    for scene in manifest["scenes"]:
        ranges = scene["ranges"]
        assert ranges["train"][1] < ranges["validation"][0] < ranges["validation"][1] < ranges["test"][0]
        assert ranges["train"][0] <= ranges["wiener_fit"][0] < ranges["wiener_fit"][1]
        assert ranges["wiener_fit"][1] < ranges["wiener_calibration"][0] < ranges["wiener_calibration"][1] <= ranges["train"][1]
        for source in scene["sources"]:
            allowed[source["path"]] = ranges["train"]
    reads = []

    def guarded_read(path, start, frames):
        a, b = allowed[str(path)]
        assert a <= start < start + frames <= b, "Validation/test samples reached bank construction!"
        reads.append((str(path), start, start + frames))
        return read_mono(path, start, frames)

    monkeypatch.setattr("wiener.calculate_wiener.read_mono", guarded_read)
    monkeypatch.setattr("data.preprocess.read_mono", guarded_read)
    bundle = build_bank(cfg)
    assert reads
    assert bundle["filters"].shape == (8, 1967)
    assert bundle["normalizer"]["source_split"] == "train"
    for split in ("train", "validation", "test"):
        ds = ANCWindows(cfg, manifest, split)
        for record in ds.records:
            start = record["observation_start"]
            assert record["split_start"] <= start
            assert start + ds.observation + ds.score <= record["split_end"]


def test_end_to_end_model_is_convex_bank_frozen_and_comparison_is_paired(tiny_config):
    cfg = tiny_config
    bundle = build_bank(cfg)
    model = ANCTransformer(cfg, bundle)
    batch = torch.randn(2, 4800)
    output = model(batch)
    assert output["filter"].shape == (2, 1967)
    assert output["weights"].shape == (2, 8)
    torch.testing.assert_close(output["weights"].sum(-1), torch.ones(2))
    assert output["weights"].min() >= 0
    torch.testing.assert_close(output["filter"], output["weights"] @ model.filter_bank)
    assert "filter_bank" not in dict(model.named_parameters())
    (-output["logits"][:, 0].mean()).backward()
    assert model.decoder.scene_embedding.weight.grad.abs().sum() > 0
    checkpoint = train(cfg)
    assert checkpoint["epoch"] == 1
    selected, loaded_bundle, _ = load_model(cfg)
    rows = evaluate(cfg, selected, loaded_bundle, "test",
                    ("transformer", "scene_wiener", "pooled_wiener", "mean_wiener"))
    assert len(rows) == 32
    for name in cfg["scene_names"]:
        scene_rows = [r for r in rows if r["scene"] == name]
        assert len({(r["observation_start"], r["control_start"], r["scored_samples"]) for r in scene_rows}) == 1
    changed = copy.deepcopy(cfg)
    changed["path_id"] = 2
    with pytest.raises(ValueError, match="Bank configuration changed"):
        verified_bundle(changed)


def test_config_is_relative_to_yaml_and_rejects_bad_v1(tiny_config, tmp_path):
    cfg = tiny_config
    assert Path(cfg["bank_file"]).is_absolute()
    source = yaml.safe_load(Path(cfg["config_file"]).read_text(encoding="utf-8"))
    source["filter_length"] = 1024
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="1967 taps"):
        load_config(invalid)
