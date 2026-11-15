"""Scientific invariants shared by SFANC calibration, labels, and inference."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from anc.anc_simulator import simulate, simulate_numpy
from anc.metrics import power_spectrum, score_signals, third_octave_band_power
from experiments.common import acoustic_mse
from models.decoder import WienerPriorTokenizer
from models.filter_head import FilterHead
from models.rope import apply_rope
from wiener.calculate_wiener import energy_diagnostics, solve_candidates, statistic_vectors


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
    assert numpy_result["anti_noise"][0] == pytest.approx(y[context])


def test_acoustic_loss_is_the_whole_clip_residual_mean_square():
    generator = torch.Generator().manual_seed(11)
    reference = torch.randn(2, 16, generator=generator, dtype=torch.float64)
    disturbance = torch.randn(2, 16, generator=generator, dtype=torch.float64)
    secondary = torch.tensor([0.7, -0.2, 0.05], dtype=torch.float64)
    coefficients = torch.randn(2, 4, generator=generator, requires_grad=True)
    result = {"filter": coefficients}
    batch = {"reference": reference, "disturbance": disturbance, "context": 0,
             "secondary_path": secondary}
    objective = acoustic_mse(result, batch)
    expected = simulate(reference, disturbance, coefficients, secondary, 0)["residual"].square().mean()
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert coefficients.grad is not None and torch.isfinite(coefficients.grad).all()
    assert coefficients.grad.abs().sum() > 0


def test_metric_known_gain_and_band_rebound():
    x = np.random.default_rng(3).normal(size=16000)
    settings = {"sample_rate": 16000, "n_fft": 2048, "hop_length": 512}
    metrics = score_signals(x, x * 0.5, x * 0, **settings)
    assert metrics["primary_nr_db"] == pytest.approx(20 * np.log10(2), abs=1e-9)
    assert metrics["broadband_nr_db"] == pytest.approx(20 * np.log10(2), abs=1e-9)
    assert metrics["maximum_rebound_db"] == 0
    amplified = score_signals(x, x * 2, x * 0, **settings)
    assert amplified["maximum_rebound_db"] == pytest.approx(20 * np.log10(2), abs=1e-9)


def test_spectrum_matches_torch_stft_scoring_convention():
    x = np.random.default_rng(17).normal(size=16000)
    waveform = torch.tensor(x, dtype=torch.float64)
    waveform = waveform - waveform.mean()
    n_fft, hop = 2048, 512
    padded = max(len(x), n_fft)
    padded += (-(padded - n_fft)) % hop
    waveform = torch.nn.functional.pad(waveform, (0, padded - len(x)))
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    spectrum = torch.stft(waveform, n_fft, hop, window=window, center=False, return_complex=True)
    expected = spectrum.abs().square().mean(-1) / window.square().sum()
    expected[1:-1] *= 2
    np.testing.assert_allclose(power_spectrum(x, n_fft, hop), expected.numpy(), rtol=1e-12, atol=1e-12)

    centres, off = third_octave_band_power(x, 16000, n_fft, hop)
    other_centres, on = third_octave_band_power(x * 0.5, 16000, n_fft, hop)
    np.testing.assert_array_equal(centres, other_centres)
    np.testing.assert_allclose(off / on, 4.0, rtol=1e-12)


def test_rope_preserves_norm_and_depends_on_temporal_position():
    query = torch.randn(2, 4, 7, 8)
    rotated = apply_rope(query)
    torch.testing.assert_close(rotated.square().sum(-1), query.square().sum(-1))
    torch.testing.assert_close(rotated[:, :, 0], query[:, :, 0])
    assert not torch.allclose(rotated[:, :, 1], query[:, :, 1])

    repeated = query[:, :, :1].expand(-1, -1, 7, -1)
    positioned = apply_rope(repeated)
    assert not torch.allclose(positioned[:, :, 0], positioned[:, :, 1])


def test_residual_filter_head_starts_at_prior_and_learns_a_delta():
    torch.manual_seed(23)
    width, taps, batch = 8, 7, 3
    decoded = torch.randn(batch, 5, width, requires_grad=True)
    prior = torch.randn(batch, taps)
    head = FilterHead(width, taps, output_mode="residual")

    initial = head(decoded, prior)
    torch.testing.assert_close(initial["filter"], prior)
    torch.testing.assert_close(initial["prior_scale"], torch.ones(batch, 1))

    with torch.no_grad():
        head.output[-1].weight.normal_(mean=0.0, std=0.05)
    adapted = head(decoded, prior)
    assert not torch.allclose(adapted["filter"], prior)
    (adapted["filter"] - prior).square().sum().backward()
    assert head.output[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(head.output[-1].weight.grad).all()


@pytest.mark.parametrize(("mode", "tokens"), [("linear", 32), ("mlp", 64), ("fft_pool", 32)])
def test_wiener_prior_tokenizer_modes_are_learnable(mode, tokens):
    torch.manual_seed(29)
    prior = torch.randn(3, 7, requires_grad=True)
    tokenizer = WienerPriorTokenizer(
        7, 8, tokens, mode=mode, normalize_input=False, hidden_dim=12, dropout=0.0,
    )
    output = tokenizer(prior)
    assert output.shape == (3, tokens, 8)
    output.square().mean().backward()
    assert prior.grad is not None and torch.isfinite(prior.grad).all()
    assert prior.grad.abs().sum() > 0


def test_direct_filter_scale_can_be_ablated_without_changing_shapes():
    decoded = torch.randn(3, 5, 8)
    prior = torch.randn(3, 7)
    head = FilterHead(8, 7, output_mode="direct", scale_mode="unit")
    output = head(decoded, prior)
    assert output["filter"].shape == (3, 7)
    torch.testing.assert_close(output["prior_scale"], torch.ones(3, 1))
