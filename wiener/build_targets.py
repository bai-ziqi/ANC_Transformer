"""Cache clip/environment Wiener labels chosen by same-clip residual MSE."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from anc.anc_simulator import causal_numpy, simulate_numpy
from data.environments import load_environments
from data.sfanc import inspect_sfanc, load_noise
from utils import bank_signature_cfg, load_config, sha256
from wiener.calculate_wiener import fit_wiener_array


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


def _validate_resume(index, expected, destination, records):
    for key in ("format_version", "manifest_sha256", "config_signature",
                "environment_signature", "filter_length", "total_records", "partial"):
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
    """Build all selected pairs without keeping the full target bank in memory."""
    if cfg.get("protocol", {}).get("mode") != "sfanc":
        raise ValueError("build_targets requires protocol.mode=sfanc.")
    manifest = inspect_sfanc(cfg, limit=limit, save=False)
    environments = load_environments(cfg)
    records = manifest["records"]
    destination = Path(cfg["targets_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "index.json"
    index = {
        "format_version": 1,
        "manifest_sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
        "config_signature": bank_signature_cfg(cfg),
        "environment_signature": _environment_signature(environments),
        "filter_length": cfg["filter_length"],
        "solver": "biased_stationary_toeplitz",
        "selection_metric": "minimum_causal_residual_mse",
        "reference_kind": "best_in_configured_ridge_candidate_grid",
        "evaluation_protocol": "same clip; zero initial states; full clip scored",
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
    for start in range(index["completed_records"], len(records), chunk_size):
        selected_records = records[start:start + chunk_size]
        filters, diagnostics = [], []
        for record in selected_records:
            try:
                reference = load_noise(record, cfg)
                environment = environments[record["environment_id"]]
                coefficients, selection, candidates = fit_wiener_array(
                    reference, environment["primary"], environment["secondary"],
                    cfg["filter_length"], ridges,
                )
                stored = coefficients.astype(np.float32)
                if not np.isfinite(stored).all():
                    raise ValueError("Selected coefficients overflowed the float32 target format.")
                disturbance = causal_numpy(reference, environment["primary"])
                outcome = simulate_numpy(reference, disturbance, stored.astype(np.float64),
                                         environment["secondary"], context=0)
                stored_mse = float(np.mean(outcome["residual"] ** 2))
                if not np.isfinite(stored_mse):
                    raise ValueError("Stored target produced a nonfinite residual.")
                filters.append(stored)
                diagnostics.append({
                    "record_id": record["record_id"], "sample_id": record["sample_id"],
                    "environment_id": record["environment_id"], "split": record["split"],
                    "selection": selection, "candidates": candidates,
                    "stored_residual_mse": stored_mse,
                    "disturbance_mse": float(np.mean(disturbance ** 2)),
                })
            except Exception as exc:
                _atomic_json(destination / "last_failure.json", {
                    "record_id": record["record_id"], "sample_id": record["sample_id"],
                    "environment_id": record["environment_id"], "path": record["path"],
                    "error_type": type(exc).__name__, "error": str(exc),
                    "completed_records": index["completed_records"],
                })
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
        print(f"Wiener targets {index['completed_records']}/{len(records)}; "
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
