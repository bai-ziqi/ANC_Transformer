"""Exact matrix-free finite-window least-squares helpers for SFANC labels."""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import correlate
from scipy.sparse.linalg import LinearOperator, lsmr

from anc.anc_simulator import causal_numpy


class FiniteWindowLSConvergenceError(RuntimeError):
    """Raised when LSMR did not produce a verified minimum-residual solution."""


def _causal_adjoint(value, filtered_reference, taps):
    """Apply the transpose of the truncated causal-convolution operator."""
    value = np.asarray(value, dtype=np.float64)
    filtered_reference = np.asarray(filtered_reference, dtype=np.float64)
    if value.ndim != 1 or filtered_reference.ndim != 1 or value.shape != filtered_reference.shape:
        raise ValueError("Adjoint operands must be matching one-dimensional signals.")
    if taps <= 0 or len(value) < taps:
        raise ValueError("Adjoint requires at least `taps` samples.")
    correlation = correlate(value, filtered_reference, mode="full", method="fft")
    start = len(value) - 1
    return correlation[start:start + taps]


def causal_window_operator(filtered_reference, taps):
    """Linear operator mapping w to conv(filtered_reference, w)[:N]."""
    reference = np.asarray(filtered_reference, dtype=np.float64)
    if reference.ndim != 1 or not reference.size or not np.isfinite(reference).all():
        raise ValueError("filtered_reference must be a nonempty finite 1-D vector.")
    if isinstance(taps, bool) or not isinstance(taps, (int, np.integer)) or taps <= 0:
        raise ValueError("taps must be a positive integer.")
    rows, columns = len(reference), int(taps)

    def matvec(coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (columns,):
            raise ValueError(f"Expected coefficients with shape ({columns},).")
        return causal_numpy(reference, coefficients)

    def rmatvec(value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (rows,):
            raise ValueError(f"Expected an input with shape ({rows},).")
        return _causal_adjoint(value, reference, columns)

    return LinearOperator((rows, columns), matvec=matvec, rmatvec=rmatvec,
                          dtype=np.float64)


def _matrix_diagnostics(filtered_reference, taps):
    """Exact Gram diagnostics; opt-in because the dense design uses O(NW) memory."""
    reference = np.asarray(filtered_reference, dtype=np.float64)
    first_column = np.concatenate((reference, np.zeros(taps - 1)))
    first_row = np.concatenate((reference[:1], np.zeros(taps - 1)))
    design = toeplitz(first_column, first_row)[:len(reference)]
    gram = design.T @ design
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = float(max(eigenvalues[-1], 0.0))
    tolerance = largest * max(design.shape) * np.finfo(np.float64).eps
    positive = eigenvalues[eigenvalues > tolerance]
    return {
        "gram_smallest_eigenvalue": float(max(eigenvalues[0], 0.0)),
        "gram_largest_eigenvalue": largest,
        "gram_numerical_rank": int(len(positive)),
        "gram_numerical_tolerance": float(tolerance),
        "design_condition_number": (float(np.sqrt(largest / positive[0]))
                                    if len(positive) else None),
    }


def solve_finite_window_ls(filtered_reference, disturbance, taps, relative_ridge=0.0,
                           anchor=None, *, matrix_diagnostics=False, maxiter=2000,
                           atol=1e-11, btol=1e-11, conlim=1e8):
    """Solve the complete scored-window LS objective with an optional prior anchor.

    The formal label path must use :func:`solve_pure_finite_window_ls`, which fixes
    ``relative_ridge=0`` and never accepts a prior anchor. This lower-level function
    remains available for explicit ridge diagnostics.

    argmin_w ||d - conv(z,w)[:N]||^2
             + lambda * ||w - anchor||^2.
    The ridge scale is relative to the mean diagonal of A.T A.
    """
    reference = np.asarray(filtered_reference, dtype=np.float64)
    target = np.asarray(disturbance, dtype=np.float64)
    if reference.shape != target.shape or reference.ndim != 1 or len(reference) <= taps:
        raise ValueError("Finite-window data must be matching 1-D vectors longer than taps.")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("Finite-window data must be finite.")
    relative_ridge = float(relative_ridge)
    if not np.isfinite(relative_ridge) or relative_ridge < 0:
        raise ValueError("relative_ridge must be finite and nonnegative.")
    if anchor is None:
        anchor = np.zeros(taps, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (taps,) or not np.isfinite(anchor).all():
        raise ValueError(f"anchor must be a finite vector with shape ({taps},).")
    if not np.isfinite(atol) or atol <= 0 or not np.isfinite(btol) or btol <= 0:
        raise ValueError("LSMR atol/btol must be finite and positive.")
    if not np.isfinite(conlim) or conlim <= 1:
        raise ValueError("LSMR conlim must be finite and greater than one.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("LSMR maxiter must be a positive integer.")

    diagonal = np.asarray([np.square(reference[:len(reference) - index]).sum()
                           for index in range(taps)], dtype=np.float64)
    absolute_ridge = relative_ridge * float(diagonal.mean())
    base = causal_window_operator(reference, taps)
    if absolute_ridge:
        sqrt_ridge = np.sqrt(absolute_ridge)

        def matvec(coefficients):
            return np.concatenate((base.matvec(coefficients), sqrt_ridge * coefficients))

        def rmatvec(value):
            return base.rmatvec(value[:len(reference)]) + sqrt_ridge * value[len(reference):]

        operator = LinearOperator((len(reference) + taps, taps), matvec=matvec,
                                  rmatvec=rmatvec, dtype=np.float64)
        right_hand_side = np.concatenate((target, sqrt_ridge * anchor))
    else:
        operator = base
        right_hand_side = target
    coefficients, istop, iterations, normr, normar, norma, condition, normx = lsmr(
        operator, right_hand_side, damp=0.0, atol=atol, btol=btol,
        conlim=conlim, maxiter=int(maxiter),
    )
    if not np.isfinite(coefficients).all():
        raise ValueError("Finite-window least squares returned nonfinite coefficients.")
    result = {
        "solver": "lsmr_exact_causal_scored_window",
        "ridge_relative_to_mean_gram_diagonal": relative_ridge,
        "absolute_ridge": absolute_ridge,
        "mean_gram_diagonal": float(diagonal.mean()),
        "lsmr_istop": int(istop),
        "lsmr_iterations": int(iterations),
        "lsmr_normr": float(normr),
        "lsmr_normar": float(normar),
        "lsmr_norma": float(norma),
        "lsmr_condition_estimate": float(condition) if np.isfinite(condition) else None,
        "lsmr_normx": float(normx),
        "lsmr_atol": float(atol),
        "lsmr_btol": float(btol),
        "lsmr_conlim": float(conlim),
        "lsmr_maxiter": int(maxiter),
        "initialization": "zero (LSMR default)",
    }
    if matrix_diagnostics:
        result.update(_matrix_diagnostics(reference, taps))
    return coefficients, result


def solve_pure_finite_window_ls(
    filtered_reference,
    disturbance,
    taps,
    *,
    maxiter=2000,
    atol=1e-11,
    btol=1e-11,
    conlim=1e8,
    normal_residual_tolerance=1e-7,
):
    """Return and verify the pure full-window least-squares label ``w``.

    No ridge regularization or prior anchor is accepted. LSMR must stop for a
    documented least-squares reason, and the normal-equation residual is recomputed
    independently from the returned coefficients before this function succeeds.
    """
    if not np.isfinite(normal_residual_tolerance) or normal_residual_tolerance <= 0:
        raise ValueError("normal_residual_tolerance must be finite and positive.")
    coefficients, diagnostics = solve_finite_window_ls(
        filtered_reference,
        disturbance,
        taps,
        relative_ridge=0.0,
        anchor=None,
        maxiter=maxiter,
        atol=atol,
        btol=btol,
        conlim=conlim,
    )
    reference = np.asarray(filtered_reference, dtype=np.float64)
    target = np.asarray(disturbance, dtype=np.float64)
    residual = target - causal_numpy(reference, coefficients)
    gradient = _causal_adjoint(residual, reference, taps)
    residual_l2 = float(np.linalg.norm(residual))
    target_l2 = float(np.linalg.norm(target))
    gradient_l2 = float(np.linalg.norm(gradient))
    operator_norm = float(diagnostics["lsmr_norma"])
    coefficient_norm = float(np.linalg.norm(coefficients))
    tiny = np.finfo(np.float64).tiny
    normal_scale = max(operator_norm * residual_l2, tiny)
    normal_relative = gradient_l2 / normal_scale if residual_l2 > 0 else 0.0
    normal_equation_backward_error = gradient_l2 / max(
        operator_norm * operator_norm * coefficient_norm +
        operator_norm * target_l2,
        tiny,
    )
    mse = float(np.mean(np.square(residual)))

    # With damp=0, these are the documented successful LSMR outcomes. Condition
    # loss, a zero solution, and the iteration limit are hard failures for labels.
    acceptable_stops = {1, 2, 4}
    stop = int(diagnostics["lsmr_istop"])
    failure = None
    if stop not in acceptable_stops:
        failure = f"LSMR stopped with istop={stop}, not a successful least-squares status"
    elif normal_equation_backward_error > normal_residual_tolerance:
        failure = (
            "recomputed normal-equation backward error is too large: "
            f"{normal_equation_backward_error:.6e} > {normal_residual_tolerance:.6e}"
        )
    if failure is not None:
        raise FiniteWindowLSConvergenceError(failure)

    diagnostics.update({
        "solver": "pure_full_window_lsmr",
        "ridge_relative_to_mean_gram_diagonal": 0.0,
        "absolute_ridge": 0.0,
        "prior_anchor_used": False,
        "acceptance": "passed",
        "acceptable_lsmr_istop": sorted(acceptable_stops),
        "normal_residual_tolerance": float(normal_residual_tolerance),
        "residual_l2": residual_l2,
        "residual_rms": float(np.sqrt(mse)),
        "residual_mse": mse,
        "residual_relative_to_disturbance": residual_l2 / max(target_l2, tiny),
        "recomputed_normal_residual_l2": gradient_l2,
        "normal_residual_relative": float(normal_relative),
        "normal_equation_backward_error": float(normal_equation_backward_error),
        "coefficient_l2": coefficient_norm,
        "coefficient_peak_abs": float(np.abs(coefficients).max()),
    })
    return coefficients, diagnostics

