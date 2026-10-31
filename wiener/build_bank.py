"""Build 8 scene FIRs and a fair pooled baseline from outer TRAIN ONLY."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from anc.anc_simulator import simulate_numpy
from anc.metrics import score_signals
from data.inspect_data import inspect, load_secondary
from data.preprocess import fit_normalizer
from utils import bank_signature_cfg, common_parser, dump_json, load_config, seed_everything
from wiener.calculate_wiener import (calibration_audio, choose_candidate, collect_statistics,
                                    energy_diagnostics, solve_candidates)


def build_bank(cfg):
    seed_everything(cfg["seed"])
    manifest, secondary = inspect(cfg), load_secondary(cfg)
    statistics, audio, provenance, filters, selections, candidates_report = [], [], [], [], [], {}
    for scene in manifest["scenes"]:
        print(f"Fitting train-only Wiener: {scene['name']}", flush=True)
        r, p, details = collect_statistics(cfg, scene, secondary)
        statistics.append((r, p))
        provenance.append({"scene": scene["name"], **details})
        pair = calibration_audio(scene, cfg)
        audio.append(pair)
        candidates = solve_candidates(r, p, cfg["wiener"]["ridge_ratios"])
        w, selection, rows = choose_candidate(candidates, [pair], secondary, cfg)
        filters.append(w)
        selections.append(selection)
        candidates_report[scene["name"]] = rows
    filters = np.stack(filters)
    pooled_candidates = solve_candidates(np.mean([r for r, _ in statistics], 0),
                                          np.mean([p for _, p in statistics], 0),
                                          cfg["wiener"]["ridge_ratios"])
    pooled, pooled_selection, pooled_rows = choose_candidate(pooled_candidates, audio, secondary, cfg)
    normalizer = fit_normalizer(cfg, manifest)
    norms = np.linalg.norm(filters, axis=1)
    cosine = (filters @ filters.T) / (norms[:, None] * norms[None, :])
    # Cross-scene NR matrix uses calibration contained entirely within train.
    transfer = []
    for x, d in audio:
        row = []
        for w in filters:
            result = simulate_numpy(x, d, w, secondary, cfg["filter_length"] - 1)
            row.append(score_signals(d[cfg["filter_length"] - 1:], result["residual"],
                                     result["control"], cfg["sample_rate"], **cfg["metrics"])["primary_nr_db"])
        transfer.append(row)
    energy = {name: energy_diagnostics(w) for name, w in zip(cfg["scene_names"], filters)}
    diagnostics = {"source_split": "train (inner calibration only)", "energy": energy,
                   "cosine_similarity": cosine.tolist(), "calibration_cross_scene_nr_db": transfer,
                   "matrix_rows": "evaluation scene", "matrix_columns": "candidate scene filter",
                   "ridge_selections": selections, "pooled_selection": pooled_selection,
                   "candidate_ridges": candidates_report, "pooled_candidate_ridges": pooled_rows,
                   "interpretation": "A scene Wiener FIR is a conditional reference, not an unconditional NR upper bound."}
    bundle = {"format_version": 1, "config_signature": bank_signature_cfg(cfg), "manifest": manifest,
              "filters": torch.from_numpy(filters), "pooled_filter": torch.from_numpy(pooled),
              "secondary_path": torch.from_numpy(secondary), "scene_names": cfg["scene_names"],
              "path_id": cfg["path_id"], "sample_rate": cfg["sample_rate"],
              "filter_length": cfg["filter_length"], "selected_ridge_ratios": [r["ridge_ratio"] for r in selections],
              "normalizer": normalizer, "fit_provenance": provenance, "diagnostics": diagnostics}
    destination = Path(cfg["bank_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    dump_json(Path(cfg["output_dir"]) / "data_manifest.json", manifest)
    dump_json(Path(cfg["output_dir"]) / "wiener_diagnostics.json", diagnostics)
    np.savez_compressed(Path(cfg["output_dir"]) / "wiener_bank_export.npz", filters=filters,
                        pooled_filter=pooled, secondary_path=secondary,
                        noise_names=np.asarray(cfg["scene_names"]),
                        cumulative_energy=np.cumsum(filters ** 2, axis=1) / (norms[:, None] ** 2))
    print(f"Saved train-only bank {tuple(filters.shape)}: {destination}", flush=True)
    return bundle


def main():
    args = common_parser(__doc__).parse_args()
    build_bank(load_config(args.config))


if __name__ == "__main__":
    main()
