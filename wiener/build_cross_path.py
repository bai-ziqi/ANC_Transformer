"""Build scene x path Wiener priors for path-disjoint experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.cross_path import inspect_cross_path, protocol_paths
from data.inspect_data import load_secondary_paths
from data.preprocess import fit_normalizer
from utils import bank_signature_cfg, dump_json, seed_everything
from wiener.calculate_wiener import (
    calibration_audio,
    choose_candidate,
    collect_statistics,
    energy_diagnostics,
    solve_candidates,
)


def _scene_for_path(scene, path_id):
    target = scene["sources"]["targets"][str(path_id)]
    return {
        "name": scene["name"],
        "sources": [scene["sources"]["raw"], target],
        "frames": scene["frames"],
        "ranges": scene["ranges"],
    }


def build_cross_path_bank(cfg):
    seed_everything(cfg["seed"])
    manifest = inspect_cross_path(cfg)
    groups = protocol_paths(cfg)
    path_ids = manifest["path_ids"]
    secondary_paths = load_secondary_paths(cfg)
    filters = []
    selections = []
    provenance = []
    candidates_report = {}
    statistics_by_path = [[] for _ in path_ids]
    calibration_by_path = [[] for _ in path_ids]

    for scene_id, scene in enumerate(manifest["scenes"]):
        scene_filters = []
        scene_selections = []
        for path_index, path_id in enumerate(path_ids):
            print(f"Fitting scene={scene['name']} path={path_id}", flush=True)
            view = _scene_for_path(scene, path_id)
            auto, cross, details = collect_statistics(cfg, view, secondary_paths[path_id - 1])
            statistics_by_path[path_index].append((auto, cross))
            pair = calibration_audio(view, cfg)
            calibration_by_path[path_index].append(pair)
            candidates = solve_candidates(auto, cross, cfg["wiener"]["ridge_ratios"])
            coefficients, selection, rows = choose_candidate(
                candidates, [pair], secondary_paths[path_id - 1], cfg
            )
            scene_filters.append(coefficients)
            scene_selections.append(selection)
            provenance.append({
                "scene": scene["name"], "path_id": path_id,
                **details, "selection": selection,
            })
            candidates_report[f"{scene['name']}|{path_id}"] = rows
        filters.append(scene_filters)
        selections.append(scene_selections)

    filters = np.stack([np.stack(row) for row in filters], axis=0)
    pooled_filters = []
    pooled_selections = []
    pooled_reports = {}
    for path_index, path_id in enumerate(path_ids):
        stats = statistics_by_path[path_index]
        auto = np.mean([value[0] for value in stats], axis=0)
        cross = np.mean([value[1] for value in stats], axis=0)
        candidates = solve_candidates(auto, cross, cfg["wiener"]["ridge_ratios"])
        coefficients, selection, rows = choose_candidate(
            candidates, calibration_by_path[path_index], secondary_paths[path_id - 1], cfg
        )
        pooled_filters.append(coefficients)
        pooled_selections.append(selection)
        pooled_reports[str(path_id)] = rows
    pooled_filters = np.stack(pooled_filters)
    bank_secondary_paths = np.stack([secondary_paths[path_id - 1] for path_id in path_ids])

    energy = {
        f"{scene['name']}|{path_id}": energy_diagnostics(filters[scene_id, path_index])
        for scene_id, scene in enumerate(manifest["scenes"])
        for path_index, path_id in enumerate(path_ids)
    }
    normalizer = fit_normalizer(cfg, manifest)
    bundle = {
        "format_version": 2,
        "config_signature": bank_signature_cfg(cfg),
        "manifest": manifest,
        "filters": torch.from_numpy(filters),
        "pooled_filters": torch.from_numpy(pooled_filters),
        "secondary_paths": torch.from_numpy(bank_secondary_paths),
        "scene_names": cfg["scene_names"],
        "path_ids": path_ids,
        "path_splits": {key: list(value) for key, value in groups.items()},
        "sample_rate": cfg["sample_rate"],
        "filter_length": cfg["filter_length"],
        "selected_ridge_ratios": selections,
        "pooled_selections": pooled_selections,
        "normalizer": normalizer,
        "fit_provenance": provenance,
        "diagnostics": {
            "source_split": "train-only per scene/path interval",
            "energy": energy, "ridge_selections": selections,
            "pooled_selections": pooled_selections,
            "candidate_ridges": candidates_report,
            "pooled_candidate_ridges": pooled_reports,
            "prior_protocol": "Each validation/test path prior is fitted only from that path's TRAIN interval; no target-window future data is used.",
        },
    }
    destination = Path(cfg["bank_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    output = Path(cfg["output_dir"])
    dump_json(output / "data_manifest.json", manifest)
    dump_json(output / "wiener_cross_path_diagnostics.json", bundle["diagnostics"])
    np.savez_compressed(
        output / "wiener_cross_path_bank_export.npz",
        filters=filters, pooled_filters=pooled_filters,
        secondary_paths=bank_secondary_paths, path_ids=np.asarray(path_ids),
        noise_names=np.asarray(cfg["scene_names"]),
    )
    print(f"Saved cross-path bank {tuple(filters.shape)}: {destination}", flush=True)
    return bundle
