"""Read-only audit of cached Toeplitz labels against finite-window least squares.

The finite-window solution is an offline, in-sample oracle. It is intended to test
the label estimator and must not be used to claim online ANC generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import correlate
from scipy.sparse.linalg import LinearOperator, lsmr

from anc.anc_simulator import causal_numpy, simulate_numpy
from anc.metrics import score_signals
from data.sfanc import SfancDataset, load_noise, load_training_bundle
from utils import common_parser, dump_json, load_config


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
                           anchor=None, *, matrix_diagnostics=False, maxiter=2000):
    """Solve the exact scored-window objective with an optional prior anchor.

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
        operator, right_hand_side, atol=1e-11, btol=1e-11, maxiter=maxiter,
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
    }
    if matrix_diagnostics:
        result.update(_matrix_diagnostics(reference, taps))
    return coefficients, result


def _score_filter(reference, disturbance, coefficients, secondary, cfg):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    outcome = simulate_numpy(reference, disturbance, coefficients, secondary, context=0)
    residual = outcome["residual"]
    metrics = score_signals(disturbance, residual, outcome["control"],
                            cfg["sample_rate"], **cfg["metrics"])
    disturbance_mse = float(np.mean(np.square(disturbance)))
    residual_mse = float(np.mean(np.square(residual)))
    return {
        "residual_mse": residual_mse,
        "disturbance_mse": disturbance_mse,
        "residual_to_disturbance_energy_ratio": residual_mse / max(disturbance_mse, 1e-30),
        "broadband_nr_db": float(metrics["broadband_nr_db"]),
        "primary_nr_db": float(metrics["primary_nr_db"]),
        "maximum_rebound_db": float(metrics["maximum_rebound_db"]),
        "control_rms": float(metrics["control_rms"]),
        "coefficient_l2": float(_stable_norm(coefficients)),
        "coefficient_peak_abs": float(np.max(np.abs(coefficients))),
    }


def _stable_norm(value):
    value = np.asarray(value, dtype=np.float64)
    scale = float(np.max(np.abs(value))) if value.size else 0.0
    return 0.0 if scale == 0 else scale * float(np.linalg.norm(value / scale))


def _cosine(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm, second_norm = _stable_norm(first), _stable_norm(second)
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return float(np.dot(first / first_norm, second / second_norm))


def _read_cached_label(bundle, global_index, record):
    root = Path(bundle["targets_dir"])
    index = bundle["targets_index"]
    ends = np.asarray([chunk["start"] + chunk["count"] for chunk in index["chunks"]])
    chunk_index = int(np.searchsorted(ends, global_index, side="right"))
    chunk = index["chunks"][chunk_index]
    row = global_index - chunk["start"]
    with np.load(root / chunk["file"], allow_pickle=False) as archive:
        filters = np.array(archive["filters"], copy=True)
        record_ids = np.array(archive["record_ids"], copy=True)
    if str(record_ids[row]) != record["record_id"]:
        raise ValueError("Cached target row does not match the manifest record.")
    diagnostics = json.loads((root / chunk["diagnostics_file"]).read_text(encoding="utf-8"))
    if row >= len(diagnostics) or diagnostics[row]["record_id"] != record["record_id"]:
        raise ValueError("Target diagnostics row does not match the manifest record.")
    return np.asarray(filters[row], dtype=np.float64), diagnostics[row]


def _candidate_summary(score, **identity):
    return {**identity, **score}


def _select_indices(dataset, requested, count):
    if requested is None:
        if count <= 0:
            raise ValueError("--count must be positive.")
        indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
        return sorted(set(int(index) for index in indices))
    indices = [int(index) for index in requested]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--indices must contain unique validation/test indices.")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("--indices contains an index outside the selected split.")
    return indices


def diagnose_labels(cfg, split="validation", indices=None, count=8,
                    ridge_ratios=None, matrix_diagnostics=False, maxiter=2000):
    """Compare cached labels, independent priors and exact finite-window solutions."""
    if split not in ("validation", "test"):
        raise ValueError("Label diagnostics are restricted to validation or test.")
    bundle = load_training_bundle(cfg)
    dataset = SfancDataset(cfg, bundle, split)
    selected_indices = _select_indices(dataset, indices, count)
    ratios = ([float(value) for value in ridge_ratios] if ridge_ratios is not None
              else [float(value) for value in cfg["wiener"]["ridge_ratios"]])
    if not ratios or len(ratios) != len(set(ratios)) or any(
            not np.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("Finite-window ridge ratios must be distinct, finite and nonnegative.")
    environment_indices = {value: index for index, value in enumerate(bundle["environment_ids"])}
    samples, serialized_filters = [], {
        "prior": [], "cached_label": [], "exact_ls": [],
    }
    for local_index in selected_indices:
        global_index, record = dataset.records[local_index]
        reference = load_noise(record, cfg)
        environment_index = environment_indices[record["environment_id"]]
        primary = bundle["primary_paths"][environment_index].double().numpy()
        secondary = bundle["secondary_paths"][environment_index].double().numpy()
        disturbance = causal_numpy(reference, primary)
        filtered_reference = causal_numpy(reference, secondary)
        prior_raw = bundle["prior_filters"][environment_index].double().numpy()
        prior_input = prior_raw.astype(np.float32).astype(np.float64)
        cached_label, cached_diagnostic = _read_cached_label(bundle, global_index, record)

        prior_scores = {
            "float64": _score_filter(reference, disturbance, prior_raw, secondary, cfg),
            "float32_quantized_as_evaluation_input": _score_filter(
                reference, disturbance, prior_input, secondary, cfg),
        }
        cached_scores = _score_filter(reference, disturbance, cached_label, secondary, cfg)
        stored_mse = float(cached_diagnostic["stored_residual_mse"])
        consistency = (cached_scores["residual_mse"] - stored_mse) / max(abs(stored_mse), 1e-30)

        exact_candidates, exact_filters = [], []
        for relative_ridge in ratios:
            try:
                coefficients, solver = solve_finite_window_ls(
                    filtered_reference, disturbance, cfg["filter_length"], relative_ridge,
                    anchor=prior_input, matrix_diagnostics=matrix_diagnostics, maxiter=maxiter,
                )
                score64 = _score_filter(reference, disturbance, coefficients, secondary, cfg)
                score32 = _score_filter(
                    reference, disturbance, coefficients.astype(np.float32), secondary, cfg)
                exact_candidates.append({
                    "ridge_ratio": float(relative_ridge), "solver": solver,
                    "float64": score64, "float32_quantized": score32,
                })
                exact_filters.append(coefficients)
            except Exception as error:
                exact_candidates.append({
                    "ridge_ratio": float(relative_ridge),
                    "failed": True, "error_type": type(error).__name__, "error": str(error),
                })
                exact_filters.append(np.full(cfg["filter_length"], np.nan, dtype=np.float64))

        successful = [candidate for candidate in exact_candidates if not candidate.get("failed")]
        best_exact = min(successful, key=lambda candidate: candidate["float64"]["residual_mse"])
        prior_candidate = _candidate_summary(
            prior_scores["float32_quantized_as_evaluation_input"], method="environment_prior")
        cached_candidate = _candidate_summary(cached_scores, method="cached_toeplitz_label")
        exact_candidate = _candidate_summary(
            best_exact["float64"], method="exact_finite_window_ls",
            ridge_ratio=best_exact["ridge_ratio"])
        best_with_prior = min((prior_candidate, cached_candidate, exact_candidate),
                              key=lambda candidate: candidate["residual_mse"])
        candidate_mse = [float(value["residual_mse"])
                         for value in cached_diagnostic.get("candidates", [])]
        diagnostic_consistency = None
        if candidate_mse:
            selected_cached_mse = float(cached_diagnostic["selection"]["residual_mse"])
            diagnostic_consistency = (selected_cached_mse - min(candidate_mse)) / max(
                abs(min(candidate_mse)), 1e-30)
        sample = {
            "split_index": int(local_index), "global_index": int(global_index),
            "sample_id": record["sample_id"], "record_id": record["record_id"],
            "environment_id": record["environment_id"], "samples": len(reference),
            "environment_prior": prior_scores,
            "cached_label": {
                "scores": cached_scores, "stored_residual_mse": stored_mse,
                "rescoring_relative_difference": float(consistency),
                "selected_ridge_ratio": cached_diagnostic["selection"]["ridge_ratio"],
                "selection_min_candidate_relative_difference": (
                    None if diagnostic_consistency is None else float(diagnostic_consistency)),
                "candidate_rows": cached_diagnostic.get("candidates", []),
            },
            "exact_finite_window_candidates": exact_candidates,
            "best_exact": exact_candidate,
            "best_with_prior_in_candidate_pool": best_with_prior,
            "comparisons": {
                "cached_over_prior_mse_ratio": (
                    cached_scores["residual_mse"] / max(prior_candidate["residual_mse"], 1e-30)),
                "exact_over_prior_mse_ratio": (
                    exact_candidate["residual_mse"] / max(prior_candidate["residual_mse"], 1e-30)),
                "exact_over_cached_mse_ratio": (
                    exact_candidate["residual_mse"] / max(cached_candidate["residual_mse"], 1e-30)),
                "cosine_exact_cached": _cosine(
                    exact_filters[ratios.index(best_exact["ridge_ratio"])], cached_label),
                "cosine_exact_prior": _cosine(
                    exact_filters[ratios.index(best_exact["ridge_ratio"])], prior_input),
            },
        }
        samples.append(sample)
        serialized_filters["prior"].append(prior_input)
        serialized_filters["cached_label"].append(cached_label)
        serialized_filters["exact_ls"].append(np.stack(exact_filters))
        print(
            f"{record['sample_id']}: prior MSE={prior_candidate['residual_mse']:.6e}; "
            f"cached={cached_candidate['residual_mse']:.6e}; "
            f"exact rho={best_exact['ridge_ratio']:.6g} "
            f"MSE={exact_candidate['residual_mse']:.6e}; best={best_with_prior['method']}",
            flush=True,
        )
    best_names = [sample["best_with_prior_in_candidate_pool"]["method"] for sample in samples]
    summary = {
        "samples": len(samples),
        "prior_beats_cached": sum(sample["cached_label"]["scores"]["residual_mse"] >
                                  sample["environment_prior"]["float32_quantized_as_evaluation_input"]["residual_mse"]
                                  for sample in samples),
        "exact_beats_cached": sum(sample["best_exact"]["residual_mse"] <
                                  sample["cached_label"]["scores"]["residual_mse"]
                                  for sample in samples),
        "exact_beats_prior": sum(sample["best_exact"]["residual_mse"] <
                                 sample["environment_prior"]["float32_quantized_as_evaluation_input"]["residual_mse"]
                                 for sample in samples),
        "best_method_counts": {name: best_names.count(name) for name in sorted(set(best_names))},
        "mean_exact_over_prior_mse_ratio": float(np.mean([
            sample["comparisons"]["exact_over_prior_mse_ratio"] for sample in samples])),
        "mean_exact_over_cached_mse_ratio": float(np.mean([
            sample["comparisons"]["exact_over_cached_mse_ratio"] for sample in samples])),
    }
    return {
        "format_version": 1,
        "scope": "read-only; same-clip exact finite-window LS is an offline in-sample oracle",
        "split": split,
        "ridge_ratios": ratios,
        "selected_split_indices": selected_indices,
        "summary": summary,
        "samples": samples,
    }, {
        "record_ids": np.asarray([sample["record_id"] for sample in samples]),
        "sample_ids": np.asarray([sample["sample_id"] for sample in samples]),
        "environment_ids": np.asarray([sample["environment_id"] for sample in samples]),
        "ridge_ratios": np.asarray(ratios, dtype=np.float64),
        "prior_filters": np.stack(serialized_filters["prior"]),
        "cached_label_filters": np.stack(serialized_filters["cached_label"]),
        "exact_ls_filters": np.stack(serialized_filters["exact_ls"]),
    }


def _parse_indices(value):
    if value is None:
        return None
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--indices must be comma-separated integers.") from error


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", type=_parse_indices,
                           help="Comma-separated indices in the selected split.")
    selection.add_argument("--count", type=int, default=None,
                           help="Deterministic evenly spaced samples (default: 8).")
    parser.add_argument("--ridge-ratios", type=lambda value: [float(item) for item in value.split(",")],
                        help="Override finite-window ridge ratios.")
    parser.add_argument("--matrix-diagnostics", action="store_true",
                        help="Build each dense design matrix to report exact Gram rank/condition.")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--output", default="label_diagnostics.json")
    args = parser.parse_args()
    if args.maxiter <= 0:
        parser.error("--maxiter must be positive")
    if args.indices is None and args.count is None:
        args.count = 8
    cfg = load_config(args.config)
    report, filters = diagnose_labels(
        cfg, split=args.split, indices=args.indices, count=args.count,
        ridge_ratios=args.ridge_ratios, matrix_diagnostics=args.matrix_diagnostics,
        maxiter=args.maxiter,
    )
    output = Path(args.output).resolve()
    dump_json(output, report)
    filter_output = output.with_suffix(".npz")
    with filter_output.open("wb") as handle:
        np.savez_compressed(handle, **filters)
    print(f"Saved label diagnostics: {output}", flush=True)
    print(f"Saved compared filters: {filter_output}", flush=True)


if __name__ == "__main__":
    main()
