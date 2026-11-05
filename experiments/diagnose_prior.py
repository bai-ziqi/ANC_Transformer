"""Audit whether the adaptive FIR generator actually uses its scene/path prior."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from anc.anc_simulator import simulate
from anc.metrics import score_signals
from experiments.common import load_model, loader_for
from utils import common_parser, dump_json, load_config


TRANSFORMER_VARIANTS = (
    "transformer_correct_prior",
    "transformer_scene_swap",
    "transformer_path_swap",
    "transformer_mean_prior",
    "transformer_pooled_prior",
)


def _alternate_path_ids(path_ids, split_paths, bank_path_ids):
    """Choose a deterministic wrong prior path without touching target audio."""
    split_paths = tuple(int(value) for value in split_paths)
    all_paths = tuple(int(value) for value in bank_path_ids)
    result = []
    for path_id in path_ids:
        choices = [value for value in split_paths if value != int(path_id)]
        if not choices:
            choices = [value for value in all_paths if value != int(path_id)]
        if not choices:
            raise ValueError("A wrong-path prior requires at least two paths.")
        result.append(choices[0])
    return result


def _aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], "ALL")].append(row)
        groups[(row["variant"], int(row["path_id"]))].append(row)
    summary = []
    for (variant, path_id), values in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        summary.append({
            "variant": variant,
            "path_id": path_id,
            "episodes": len(values),
            "mean_primary_nr_db": float(np.mean([v["primary_nr_db"] for v in values])),
            "mean_broadband_nr_db": float(np.mean([v["broadband_nr_db"] for v in values])),
            "mean_residual_mse": float(np.mean([v["residual_mse"] for v in values])),
            "maximum_rebound_db": float(max(v["maximum_rebound_db"] for v in values)),
            "mean_relative_filter_mse": float(np.mean([v["relative_filter_mse"] for v in values])),
            "mean_output_sensitivity_l2": float(np.mean([v["output_sensitivity_l2"] for v in values])),
            "mean_output_sensitivity_cosine": float(np.mean([v["output_sensitivity_cosine"] for v in values])),
        })
    return summary


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def diagnose(cfg, split="validation"):
    model, bundle, checkpoint = load_model(cfg)
    if bundle.get("format_version") != 2:
        raise ValueError("Prior diagnostics require a cross-path bank.")
    model.eval()
    device = next(model.parameters()).device
    bank = bundle["filters"].to(device)
    pooled = bundle["pooled_filters"].to(device)
    bank_path_ids = [int(value) for value in bundle["path_ids"]]
    split_paths = cfg["protocol"][f"{split}_paths"]
    rows = []

    for batch in loader_for(cfg, bundle, split):
        observation = batch["observation"].to(device)
        reference = batch["reference"].to(device)
        disturbance = batch["disturbance"].to(device)
        secondary = batch["secondary_path"].to(device)
        scene_ids = batch["scene_id"].to(device)
        path_ids = [int(value) for value in batch["path_id"]]
        context = int(batch["context"][0])

        path_indices = torch.tensor(
            [bank_path_ids.index(path_id) for path_id in path_ids],
            device=device,
            dtype=torch.long,
        )
        correct_prior = batch["prior_filter"].to(device).double()
        swapped_scene_ids = (scene_ids + 1) % len(cfg["scene_names"])
        swapped_scene_prior = bank[swapped_scene_ids, path_indices].double()

        swapped_path_ids = _alternate_path_ids(path_ids, split_paths, bank_path_ids)
        swapped_path_indices = torch.tensor(
            [bank_path_ids.index(path_id) for path_id in swapped_path_ids],
            device=device,
            dtype=torch.long,
        )
        swapped_path_prior = bank[scene_ids, swapped_path_indices].double()
        mean_prior = bank.mean(dim=0)[path_indices].double()
        pooled_prior = pooled[path_indices].double()

        predictions = {
            "transformer_correct_prior": model(observation, correct_prior)["filter"].double(),
            "transformer_scene_swap": model(observation, swapped_scene_prior)["filter"].double(),
            "transformer_path_swap": model(observation, swapped_path_prior)["filter"].double(),
            "transformer_mean_prior": model(observation, mean_prior)["filter"].double(),
            "transformer_pooled_prior": model(observation, pooled_prior)["filter"].double(),
            "prior_correct": correct_prior,
        }

        denominator = correct_prior.square().sum(-1).clamp_min(1e-12)
        correct_norm = correct_prior.norm(dim=-1).clamp_min(1e-12)
        correct_model_filter = predictions["transformer_correct_prior"]
        correct_model_norm = correct_model_filter.norm(dim=-1).clamp_min(1e-12)

        for variant, coefficients in predictions.items():
            filter_delta = coefficients - correct_prior
            relative_filter_mse = filter_delta.square().sum(-1) / denominator
            prior_cosine = (coefficients * correct_prior).sum(-1) / (
                coefficients.norm(dim=-1).clamp_min(1e-12) * correct_norm
            )
            model_delta = coefficients - correct_model_filter
            output_sensitivity = model_delta.norm(dim=-1) / correct_model_norm
            output_sensitivity_cosine = (coefficients * correct_model_filter).sum(-1) / (
                coefficients.norm(dim=-1).clamp_min(1e-12) * correct_model_norm
            )
            outcome = simulate(reference, disturbance, coefficients, secondary, context)
            residual = outcome["residual"].cpu().numpy()
            control = outcome["control"].cpu().numpy()
            target = disturbance[:, context:].cpu().numpy()
            for i in range(len(scene_ids)):
                metrics = score_signals(
                    target[i], residual[i], control[i], cfg["sample_rate"], **cfg["metrics"]
                )
                rows.append({
                    "variant": variant,
                    "scene": batch["scene_name"][i],
                    "scene_id": int(scene_ids[i]),
                    "path_id": path_ids[i],
                    "observation_start": int(batch["observation_start"][i]),
                    "residual_mse": float(np.mean(residual[i] ** 2)),
                    "relative_filter_mse": float(relative_filter_mse[i]),
                    "prior_cosine": float(prior_cosine[i]),
                    "output_sensitivity_l2": float(output_sensitivity[i]),
                    "output_sensitivity_cosine": float(output_sensitivity_cosine[i]),
                    **metrics,
                })

    summary = _aggregate(rows)
    output_root = Path(cfg["output_dir"])
    _write_csv(output_root / f"prior_diagnostics_{split}_episodes.csv", rows)
    _write_csv(output_root / f"prior_diagnostics_{split}_summary.csv", summary)
    dump_json(output_root / f"prior_diagnostics_{split}.json", {
        "summary": summary,
        "split": split,
        "checkpoint_epoch": checkpoint["epoch"],
        "bank_sha256": checkpoint["bank_sha256"],
        "interpretation": (
            "Compare transformer_correct_prior with scene/path-swapped and mean-prior variants. "
            "Large NR changes show prior usefulness; high output_sensitivity_l2 means the FIR "
            "changes when the prior changes; low sensitivity with unchanged NR indicates prior neglect."
        ),
    })
    for row in summary:
        if row["path_id"] == "ALL":
            print(
                f'{row["variant"]}: NR={row["mean_primary_nr_db"]:.3f} dB; '
                f'relative filter MSE={row["mean_relative_filter_mse"]:.4f}; '
                f'output sensitivity={row["mean_output_sensitivity_l2"]:.4f}; '
                f'output cosine={row["mean_output_sensitivity_cosine"]:.4f}'
            )
    print(f"Saved prior diagnostics under {output_root}")
    return summary


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()
    diagnose(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
