"""Cache verified pure finite-window least-squares labels for each clip."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from anc.anc_simulator import causal_numpy, simulate_numpy
from data.environments import load_environments
from data.sfanc import inspect_sfanc, load_noise
from utils import bank_signature_cfg, load_config, load_torch, sha256
from wiener.calculate_wiener import fit_wiener_array
from wiener.finite_window_ls import (
    FiniteWindowLSConvergenceError,
    solve_pure_finite_window_ls,
)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _environment_signature(environments):
    return {
        name: {"sample_rate": value["sample_rate"],
               "primary_sha256": value["provenance"]["primary"]["sha256"],
               "secondary_sha256": value["provenance"]["secondary"]["sha256"]}
        for name, value in environments.items()
    }


def _least_squares_settings(cfg):
    settings = cfg["wiener"].get("least_squares", {})
    values = {
        "atol": float(settings.get("atol", 1e-10)),
        "btol": float(settings.get("btol", 1e-10)),
        "conlim": float(settings.get("conlim", 1e8)),
        "maxiter": int(settings.get("maxiter", 2000)),
        "normal_residual_tolerance": float(
            settings.get("normal_residual_tolerance", 1e-7)
        ),
        "max_quantization_relative_increase": (
            None if settings.get("max_quantization_relative_increase") is None
            else float(settings["max_quantization_relative_increase"])
        ),
        "control_comparison_relative_tolerance": float(
            settings.get("control_comparison_relative_tolerance", 1e-8)
        ),
    }
    if not np.isfinite(values["atol"]) or values["atol"] <= 0:
        raise ValueError("wiener.least_squares.atol must be finite and positive.")
    if not np.isfinite(values["btol"]) or values["btol"] <= 0:
        raise ValueError("wiener.least_squares.btol must be finite and positive.")
    if not np.isfinite(values["conlim"]) or values["conlim"] <= 1:
        raise ValueError("wiener.least_squares.conlim must be finite and greater than one.")
    if values["maxiter"] <= 0:
        raise ValueError("wiener.least_squares.maxiter must be positive.")
    for key in ("normal_residual_tolerance", "control_comparison_relative_tolerance"):
        if not np.isfinite(values[key]) or values[key] < 0:
            raise ValueError(f"wiener.least_squares.{key} must be finite and nonnegative.")
    quantization_limit = values["max_quantization_relative_increase"]
    if quantization_limit is not None and (
            not np.isfinite(quantization_limit) or quantization_limit < 0):
        raise ValueError(
            "wiener.least_squares.max_quantization_relative_increase must be null "
            "or finite and nonnegative."
        )
    return values


def _filter_scores(reference, disturbance, coefficients, secondary):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    outcome = simulate_numpy(reference, disturbance, coefficients, secondary, context=0)
    residual = outcome["residual"]
    control = outcome["control"]
    return {
        "residual_mse": float(np.mean(residual ** 2)),
        "disturbance_mse": float(np.mean(disturbance ** 2)),
        "control_rms": float(np.sqrt(np.mean(control ** 2))),
        "control_peak_abs": float(np.max(np.abs(control))),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "coefficient_peak_abs": float(np.max(np.abs(coefficients))),
    }


def _relative_difference(value, control):
    return float((value - control) / max(abs(control), 1e-30))


def _validate_resume(index, expected, destination, records):
    for key in ("format_version", "manifest_sha256", "config_signature",
                "environment_signature", "filter_length", "solver",
                "selection_metric", "reference_kind", "solver_config",
                "compare_toeplitz", "evaluation_protocol", "total_records", "partial"):
        if index.get(key) != expected[key]:
            raise ValueError(f"Target cache differs at {key}; choose a new targets_dir.")
    completed = 0
    for chunk in index.get("chunks", []):
        if chunk["start"] != completed or chunk["count"] <= 0:
            raise ValueError("Target cache has noncontiguous chunk ranges.")
        path = destination / chunk["file"]
        diagnostics = destination / chunk["diagnostics_file"]
        if sha256(path) != chunk["sha256"] or sha256(diagnostics) != chunk["diagnostics_sha256"]:
            raise ValueError(f"Target cache chunk was changed or corrupted: {path}.")
        with np.load(path, allow_pickle=False) as archive:
            filters = archive["filters"]
            ids = archive["record_ids"].tolist()
            selected = records[completed:completed + chunk["count"]]
            if (filters.shape != (chunk["count"], expected["filter_length"])
                    or filters.dtype != np.float32 or not np.isfinite(filters).all()
                    or ids != [record["record_id"] for record in selected]):
                raise ValueError(f"Target cache contents do not match the manifest: {path}.")
        completed += chunk["count"]
    if completed != index.get("completed_records") or completed > len(records):
        raise ValueError("Target cache completion count is invalid.")
    if bool(index.get("complete")) != (completed == len(records)):
        raise ValueError("Target cache complete flag does not match its chunks.")
    return completed


def build_targets(cfg, limit=None, resume=False):
    """Build pure full-window LS targets without keeping the full bank in memory."""
    if cfg.get("protocol", {}).get("mode") != "sfanc":
        raise ValueError("build_targets requires protocol.mode=sfanc.")
    manifest = inspect_sfanc(cfg, limit=limit, save=False)
    environments = load_environments(cfg)
    prior_bank = load_torch(cfg["bank_file"])
    environment_ids = list(environments)
    if (prior_bank.get("format_version") != 3
            or prior_bank.get("config_signature") != bank_signature_cfg(cfg)
            or prior_bank.get("environment_ids") != environment_ids):
        raise ValueError("Environment priors do not match the label configuration; rebuild priors.")
    if (not isinstance(prior_bank.get("prior_filters"), np.ndarray)
            and not hasattr(prior_bank.get("prior_filters"), "detach")):
        raise ValueError("Environment prior bank has no prior_filters tensor.")
    prior_filters = prior_bank["prior_filters"].double().numpy()
    if prior_filters.shape != (len(environment_ids), cfg["filter_length"]):
        raise ValueError("Environment prior bank has an invalid prior_filters shape.")
    least_squares = _least_squares_settings(cfg)
    compare_toeplitz = bool(cfg["wiener"].get("compare_toeplitz", True))
    records = manifest["records"]
    destination = Path(cfg["targets_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "index.json"
    index = {
        "format_version": 2,
        "manifest_sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
        "config_signature": bank_signature_cfg(cfg),
        "environment_signature": _environment_signature(environments),
        "filter_length": cfg["filter_length"],
        "solver": "pure_full_window_lsmr",
        "selection_metric": "minimum_full_window_causal_residual_mse",
        "reference_kind": "verified_finite_window_least_squares",
        "solver_config": least_squares,
        "compare_toeplitz": compare_toeplitz,
        "evaluation_protocol": "same clip; fixed 1024-tap FIR; causal zero initial state; full clip fitted and scored",
        "partial": manifest["partial"],
        "total_records": len(records),
        "completed_records": 0,
        "complete": False,
        "chunks": [],
    }
    if index_path.exists():
        if not resume:
            raise FileExistsError(f"Target cache exists: {index_path}. Use --resume or a new targets_dir.")
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        _validate_resume(previous, index, destination, records)
        index = previous
    _atomic_json(cfg["manifest_file"], manifest)
    if index["complete"]:
        print(f"Target cache is already complete: {index_path}", flush=True)
        return index
    _atomic_json(index_path, index)
    chunk_size = cfg["wiener"].get("target_chunk_size", 256)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("wiener.target_chunk_size must be a positive integer.")
    ridges = cfg["wiener"]["ridge_ratios"]
    environment_indices = {value: index for index, value in enumerate(environment_ids)}
    for start in range(index["completed_records"], len(records), chunk_size):
        selected_records = records[start:start + chunk_size]
        filters, diagnostics = [], []
        for record in selected_records:
            try:
                reference = load_noise(record, cfg)
                environment = environments[record["environment_id"]]
                primary = environment["primary"]
                secondary = environment["secondary"]
                disturbance = causal_numpy(reference, primary)
                filtered_reference = causal_numpy(reference, secondary)
                coefficients, least_squares_diagnostics = solve_pure_finite_window_ls(
                    filtered_reference,
                    disturbance,
                    cfg["filter_length"],
                    maxiter=least_squares["maxiter"],
                    atol=least_squares["atol"],
                    btol=least_squares["btol"],
                    conlim=least_squares["conlim"],
                    normal_residual_tolerance=least_squares["normal_residual_tolerance"],
                )
                stored = coefficients.astype(np.float32)
                if not np.isfinite(stored).all():
                    raise ValueError("Selected coefficients overflowed the float32 target format.")
                floating_scores = _filter_scores(
                    reference, disturbance, coefficients, secondary
                )
                stored_scores = _filter_scores(
                    reference, disturbance, stored.astype(np.float64), secondary
                )
                stored_mse = stored_scores["residual_mse"]
                if not np.isfinite(stored_mse):
                    raise ValueError("Stored target produced a nonfinite residual.")
                quantization_increase = _relative_difference(
                    stored_mse, floating_scores["residual_mse"]
                )
                quantization_limit = least_squares["max_quantization_relative_increase"]
                if (quantization_limit is not None
                        and quantization_increase > quantization_limit):
                    raise ValueError(
                        "Float32 target quantization increased residual MSE by "
                        f"{quantization_increase:.6e}, above the allowed "
                        f"{least_squares['max_quantization_relative_increase']:.6e}."
                    )

                environment_index = environment_indices[record["environment_id"]]
                prior64 = prior_filters[environment_index]
                prior32 = prior64.astype(np.float32).astype(np.float64)
                prior_scores = {
                    "float64": _filter_scores(reference, disturbance, prior64, secondary),
                    "float32_quantized": _filter_scores(
                        reference, disturbance, prior32, secondary
                    ),
                }
                prior_relative_difference = _relative_difference(
                    floating_scores["residual_mse"],
                    prior_scores["float64"]["residual_mse"],
                )
                if prior_relative_difference > least_squares["control_comparison_relative_tolerance"]:
                    raise ValueError(
                        "Pure least-squares residual MSE exceeds the same-clip environment prior by "
                        f"{prior_relative_difference:.6e} relative; check operator/score consistency."
                    )

                toeplitz_comparison = None
                if compare_toeplitz:
                    toeplitz_coefficients, toeplitz_selection, toeplitz_candidates = fit_wiener_array(
                        reference, primary, secondary, cfg["filter_length"], ridges,
                    )
                    toeplitz_scores = _filter_scores(
                        reference, disturbance, toeplitz_coefficients, secondary
                    )
                    toeplitz_relative_difference = _relative_difference(
                        floating_scores["residual_mse"], toeplitz_scores["residual_mse"]
                    )
                    if (toeplitz_relative_difference
                            > least_squares["control_comparison_relative_tolerance"]):
                        raise ValueError(
                            "Pure least-squares residual MSE exceeds the old Toeplitz candidate by "
                            f"{toeplitz_relative_difference:.6e} relative; check LSMR convergence."
                        )
                    toeplitz_comparison = {
                        "selection": toeplitz_selection,
                        "candidates": toeplitz_candidates,
                        "float64_scores": toeplitz_scores,
                        "pure_ls_minus_control_relative": toeplitz_relative_difference,
                    }
                filters.append(stored)
                diagnostics.append({
                    "record_id": record["record_id"], "sample_id": record["sample_id"],
                    "environment_id": record["environment_id"], "split": record["split"],
                    "selection": {
                        "solver": "pure_full_window_lsmr",
                        "residual_mse": floating_scores["residual_mse"],
                        "ridge_ratio": None,
                        "selection_metric": "minimum_full_window_causal_residual_mse",
                    },
                    "least_squares": least_squares_diagnostics,
                    "float64_scores": floating_scores,
                    "float32_scores": stored_scores,
                    "quantization_relative_increase": quantization_increase,
                    "environment_prior_comparison": {
                        "scores": prior_scores,
                        "pure_ls_minus_control_relative": prior_relative_difference,
                    },
                    "toeplitz_comparison": toeplitz_comparison,
                    "stored_residual_mse": stored_mse,
                    "disturbance_mse": float(np.mean(disturbance ** 2)),
                })
            except Exception as exc:
                failure = {
                    "record_id": record["record_id"], "sample_id": record["sample_id"],
                    "environment_id": record["environment_id"], "path": record["path"],
                    "error_type": type(exc).__name__, "error": str(exc),
                    "completed_records": index["completed_records"],
                }
                if isinstance(exc, FiniteWindowLSConvergenceError):
                    failure["solver_diagnostics"] = exc.diagnostics
                _atomic_json(destination / "last_failure.json", failure)
                raise RuntimeError(
                    f"Wiener target failed for {record['sample_id']} in "
                    f"{record['environment_id']}; no samples were silently skipped."
                ) from exc
        number = len(index["chunks"])
        chunk_path = destination / f"chunk_{number:06d}.npz"
        temporary = chunk_path.with_name(chunk_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, filters=np.stack(filters),
                                record_ids=np.asarray([record["record_id"] for record in selected_records]))
        temporary.replace(chunk_path)
        diagnostic_path = chunk_path.with_suffix(".json")
        _atomic_json(diagnostic_path, diagnostics)
        index["chunks"].append({
            "file": chunk_path.name, "diagnostics_file": diagnostic_path.name,
            "start": start, "count": len(selected_records),
            "sha256": sha256(chunk_path), "diagnostics_sha256": sha256(diagnostic_path),
        })
        index["completed_records"] = start + len(selected_records)
        index["complete"] = index["completed_records"] == len(records)
        _atomic_json(index_path, index)
        print(f"Finite-window LS targets {index['completed_records']}/{len(records)}; "
              f"partial={index['partial']}", flush=True)
    return index


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="Source clips per split; creates a marked partial cache.")
    parser.add_argument("--resume", action="store_true", help="Validate completed chunks and continue.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_targets(cfg, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
