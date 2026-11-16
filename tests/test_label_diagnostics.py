"""Exact finite-window label diagnostic solver invariants."""
from __future__ import annotations

import numpy as np
import pytest

from anc.anc_simulator import causal_numpy
from wiener.diagnose_labels import causal_window_operator, solve_finite_window_ls
from wiener.finite_window_ls import (
    FiniteWindowLSConvergenceError,
    solve_pure_finite_window_ls,
)


def test_causal_window_operator_adjoint_matches_matrix_transpose():
    rng = np.random.default_rng(61)
    reference = rng.normal(size=257)
    operator = causal_window_operator(reference, 31)
    coefficients = rng.normal(size=31)
    value = rng.normal(size=257)
    assert np.dot(operator.matvec(coefficients), value) == pytest.approx(
        np.dot(coefficients, operator.rmatvec(value)), rel=1e-12, abs=1e-12)


def test_finite_window_ls_recovers_a_realizable_filter():
    rng = np.random.default_rng(67)
    taps, samples = 24, 640
    filtered_reference = rng.normal(size=samples)
    expected = rng.normal(scale=0.2, size=taps)
    disturbance = causal_numpy(filtered_reference, expected)
    actual, diagnostics = solve_finite_window_ls(
        filtered_reference, disturbance, taps, relative_ridge=0.0,
        matrix_diagnostics=True,
    )
    scores = causal_numpy(filtered_reference, actual) - disturbance
    assert np.mean(np.square(scores)) < 1e-20
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert diagnostics["gram_numerical_rank"] == taps
    assert diagnostics["design_condition_number"] is not None


def test_pure_finite_window_ls_has_no_prior_term_and_checks_convergence():
    rng = np.random.default_rng(69)
    taps, samples = 32, 1024
    filtered_reference = rng.normal(size=samples)
    expected = rng.normal(scale=0.2, size=taps)
    disturbance = causal_numpy(filtered_reference, expected)
    actual, diagnostics = solve_pure_finite_window_ls(
        filtered_reference,
        disturbance,
        taps,
        atol=1e-12,
        btol=1e-12,
        normal_residual_tolerance=1e-9,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert diagnostics["prior_anchor_used"] is False
    assert diagnostics["absolute_ridge"] == 0.0
    assert diagnostics["acceptance"] == "passed"
    assert diagnostics["normal_equation_backward_error"] <= 1e-9


def test_pure_finite_window_ls_rejects_iteration_limit():
    rng = np.random.default_rng(70)
    filtered_reference = rng.normal(size=512)
    disturbance = rng.normal(size=512)
    with pytest.raises(FiniteWindowLSConvergenceError, match="istop=7"):
        solve_pure_finite_window_ls(
            filtered_reference,
            disturbance,
            64,
            maxiter=1,
            normal_residual_tolerance=1e-9,
        )


def test_causal_operator_adjoint_at_production_shape():
    rng = np.random.default_rng(68)
    reference = rng.normal(size=16000)
    operator = causal_window_operator(reference, 1024)
    coefficients = rng.normal(size=1024)
    value = rng.normal(size=16000)
    assert np.dot(operator.matvec(coefficients), value) == pytest.approx(
        np.dot(coefficients, operator.rmatvec(value)), rel=1e-11, abs=1e-9)


def test_pure_ls_recovers_filter_at_production_shape():
    rng = np.random.default_rng(72)
    filtered_reference = rng.normal(size=16000)
    expected = rng.normal(scale=0.01, size=1024)
    disturbance = causal_numpy(filtered_reference, expected)
    actual, diagnostics = solve_pure_finite_window_ls(
        filtered_reference,
        disturbance,
        1024,
        atol=1e-10,
        btol=1e-10,
        normal_residual_tolerance=1e-7,
    )
    assert diagnostics["acceptance"] == "passed"
    assert diagnostics["residual_mse"] < 1e-16
    assert np.linalg.norm(actual - expected) < 1e-7


def test_strong_ridge_anchors_the_finite_window_solution():
    rng = np.random.default_rng(71)
    taps, samples = 18, 384
    filtered_reference = rng.normal(size=samples)
    disturbance = rng.normal(size=samples)
    anchor = rng.normal(size=taps)
    actual, diagnostics = solve_finite_window_ls(
        filtered_reference, disturbance, taps, relative_ridge=1e8, anchor=anchor,
    )
    np.testing.assert_allclose(actual, anchor, atol=1e-8)
    assert diagnostics["absolute_ridge"] > 0
