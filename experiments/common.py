"""Provenance validation, losses, and identical evaluation episodes for baselines."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anc.anc_simulator import simulate
from anc.metrics import score_signals
from data.cross_path import CrossPathWindows, inspect_cross_path
from data.dataset import ANCWindows
from data.inspect_data import inspect
from models.anc_transformer import ANCTransformer
from utils import bank_signature_cfg, device_for, dump_json, load_torch, sha256


def verified_bundle(cfg):
    bundle = load_torch(cfg["bank_file"])
    version = bundle.get("format_version")
    if version not in (1, 2) or bundle["config_signature"] != bank_signature_cfg(cfg):
        raise ValueError("Bank configuration changed. Rebuild bank for this config.")
    if version == 1:
        if bundle["manifest"] != inspect(cfg):
            raise ValueError("Dataset metadata/splits changed since bank creation. Rebuild bank.")
        expected = (8, cfg["filter_length"])
        if tuple(bundle["filters"].shape) != expected or not torch.isfinite(bundle["filters"]).all():
            raise ValueError("Invalid Wiener bank shape/coefficients.")
    else:
        if bundle["manifest"] != inspect_cross_path(cfg):
            raise ValueError("Dataset metadata/splits changed since bank creation. Rebuild bank.")
        path_ids = [int(value) for value in bundle.get("path_ids", [])]
        expected = (8, len(path_ids), cfg["filter_length"])
        if not path_ids or path_ids != list(bundle["manifest"]["path_ids"]):
            raise ValueError("Cross-path bank has invalid path ids.")
        for key, shape in (
            ("filters", expected),
            ("pooled_filters", (len(path_ids), cfg["filter_length"])),
            ("secondary_paths", (len(path_ids), cfg["filter_length"])),
        ):
            value = bundle.get(key)
            if value is None or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid cross-path {key} shape/coefficients.")
    return bundle


def acoustic_mse(result, batch, secondary=None):
    """Differentiable mean squared error over the future control interval."""
    reference = batch["reference"].to(result["filter"].device, dtype=torch.float64)
    disturbance = batch["disturbance"].to(result["filter"].device, dtype=torch.float64)
    coefficients = result["filter"].to(torch.float64)
    context_value = batch["context"]
    context = int(context_value if isinstance(context_value, int) else context_value[0])
    if secondary is None:
        secondary = batch.get("secondary_path")
        if secondary is None:
            raise ValueError("secondary path is required for acoustic loss.")
    secondary = secondary.to(coefficients.device, dtype=torch.float64)
    outcome = simulate(reference, disturbance, coefficients, secondary, context)
    return outcome["residual"].square().mean()


def losses(result, cfg, batch, secondary=None):
    target = result.get("prior_filter", batch.get("prior_filter"))
    if target is None:
        relative_mse = result["filter"].new_zeros(())
    else:
        target = target.to(result["filter"].device, dtype=result["filter"].dtype)
        relative_mse = ((result["filter"] - target).square().sum(-1) /
                        target.square().sum(-1).clamp_min(1e-12)).mean()
    acoustic = acoustic_mse(result, batch, secondary)
    mode = cfg["training"]["loss_mode"]
    filter_weight = cfg["training"]["filter_loss_weight"]
    acoustic_weight = cfg["training"]["acoustic_loss_weight"]
    values = {"acoustic_mse": acoustic, "relative_filter_mse": relative_mse}
    if mode in ("filter", "combined"):
        values["weighted_filter_loss"] = filter_weight * relative_mse
    if mode in ("acoustic", "combined"):
        values["weighted_acoustic_loss"] = acoustic_weight * acoustic
    values["loss"] = sum(value for key, value in values.items() if key.endswith("_loss"))
    return values


def prior_for(model, bank, labels, batch):
    """Return one broadband prior for every sample; never pool priors."""
    if getattr(model, "architecture", None) != "prior_adaptive":
        return None
    if "prior_filter" in batch:
        return batch["prior_filter"]
    # Compatibility bridge for the current 8-scene dataset: the row's scene
    # selects its own broadband prior. The model itself has no 8-way assumption.
    return bank[labels]


def loader_for(cfg, bundle, split, shuffle=False):
    if bundle["format_version"] == 2:
        dataset = CrossPathWindows(
            cfg, bundle["manifest"], split,
            bundle["filters"], bundle["secondary_paths"],
        )
    else:
        dataset = ANCWindows(cfg, bundle["manifest"], split)
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["training"]["num_workers"], generator=generator)


def load_model(cfg, checkpoint_file=None):
    bundle = verified_bundle(cfg)
    checkpoint = load_torch(checkpoint_file or cfg["checkpoint_file"])
    if checkpoint.get("format_version") != 4:
        raise ValueError("Checkpoint predates direct-FIR/cross-path training. Retrain it.")
    if checkpoint["bank_sha256"] != sha256(cfg["bank_file"]):
        raise ValueError("Checkpoint belongs to a different bank. Retrain using this bank.")
    saved = checkpoint["config"]
    if saved["model"] != cfg["model"] or saved["dataset"]["control_seconds"] != cfg["dataset"]["control_seconds"]:
        raise ValueError("Model or future-control protocol differs from training checkpoint.")
    for key in ("loss_mode", "acoustic_loss_weight", "filter_loss_weight"):
        if saved["training"].get(key) != cfg["training"].get(key):
            raise ValueError(f"Training objective differs at {key}. Retrain the checkpoint.")
    model = ANCTransformer(cfg, bundle)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device_for(cfg["training"]["device"])).eval(), bundle, checkpoint


@torch.no_grad()
def evaluate(cfg, model, bundle, split, methods=("transformer",)):
    allowed = {"transformer", "scene_wiener", "pooled_wiener", "mean_wiener"}
    if not methods or not set(methods).issubset(allowed):
        raise ValueError("Unknown/empty baseline list.")
    model.eval()
    device = next(model.parameters()).device
    bank = bundle["filters"].to(device)
    cross_path = bundle["format_version"] == 2
    if cross_path:
        pooled = bundle["pooled_filters"].to(device)
        bank_path_ids = [int(value) for value in bundle["path_ids"]]
    else:
        secondary = bundle["secondary_path"].to(device)
        pooled = bundle["pooled_filter"].to(device)
    rows = []
    for batch in loader_for(cfg, bundle, split):
        labels = batch["scene_id"].to(device)
        prior_filter = prior_for(model, bank, labels, batch)
        prediction = model(batch["observation"].to(device), prior_filter=prior_filter)
        reference = batch["reference"].to(device)
        disturbance = batch["disturbance"].to(device)
        context = int(batch["context"][0])
        path_ids = batch["path_id"].to(device) if cross_path else torch.full_like(labels, cfg["path_id"])
        if cross_path:
            path_indices = torch.tensor([bank_path_ids.index(int(v)) for v in path_ids.cpu()],
                                        device=device, dtype=torch.long)
            scene_prior = prior_filter.double()
            pooled_batch = pooled[path_indices]
            mean_batch = bank.mean(dim=0)[path_indices]
            batch_secondary = batch["secondary_path"].to(device)
        else:
            scene_prior = bank[labels]
            pooled_batch = pooled[None].expand(len(labels), -1)
            mean_batch = bank.mean(0)[None].expand(len(labels), -1)
            batch_secondary = secondary
        candidates = {"transformer": prediction["filter"].double(),
                      "scene_wiener": scene_prior.double(),
                      "pooled_wiener": pooled_batch.double(),
                      "mean_wiener": mean_batch.double()}
        for method in methods:
            outcome = simulate(reference, disturbance, candidates[method], batch_secondary, context)
            residual, control = outcome["residual"].cpu().numpy(), outcome["control"].cpu().numpy()
            target = disturbance[:, context:].cpu().numpy()
            for i in range(len(labels)):
                metrics = score_signals(target[i], residual[i], control[i], cfg["sample_rate"], **cfg["metrics"])
                row = {"method": method, "scene": batch["scene_name"][i],
                       "observation_start": int(batch["observation_start"][i]),
                       "control_start": int(batch["control_start"][i]), "scored_samples": len(target[i]),
                       "path_id": int(path_ids[i]),
                       "residual_mse": float(np.mean(residual[i] ** 2)),
                       "disturbance_mse": float(np.mean(target[i] ** 2)), **metrics}
                if method == "transformer":
                    target_prior = scene_prior[i]
                    row["relative_filter_mse"] = float(
                        (candidates[method][i] - target_prior).square().sum() /
                        target_prior.square().sum().clamp_min(1e-12).cpu()
                    )
                rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        method, scene, path = row["method"], row["scene"], int(row["path_id"])
        for group in ((scene, path), (scene, "ALL"), ("ALL", path), ("ALL", "ALL")):
            groups[(method, *group)].append(row)
    summaries = []
    for (method, scene, path), values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        item = {"method": method, "scene": scene, "path_id": path, "episodes": len(values),
                "mean_primary_nr_db": float(np.mean([v["primary_nr_db"] for v in values])),
                "minimum_primary_nr_db": float(min(v["primary_nr_db"] for v in values)),
                "mean_broadband_nr_db": float(np.mean([v["broadband_nr_db"] for v in values])),
                "mean_residual_mse": float(np.mean([v["residual_mse"] for v in values])),
                "mean_episode_maximum_rebound_db": float(np.mean([v["maximum_rebound_db"] for v in values])),
                "maximum_rebound_db": float(max(v["maximum_rebound_db"] for v in values)),
                "mean_positive_band_rebound_db": float(np.mean([v["mean_positive_band_rebound_db"] for v in values])),
                "maximum_control_peak_abs": float(max(v["control_peak_abs"] for v in values))}
        if method == "transformer":
            item["relative_filter_mse"] = float(np.mean([v["relative_filter_mse"] for v in values]))
        summaries.append(item)
    return summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError("No results to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation(cfg, rows, split, prefix, checkpoint):
    root = Path(cfg["output_dir"])
    summary = summarize(rows)
    protocol_mode = cfg.get("protocol", {}).get("mode", "single_path")
    if protocol_mode == "cross_path":
        scope = "Eight known scenes; disjoint acoustic paths with the SAME source recordings; no unseen-scene or cross-recording claim."
    else:
        scope = "Eight known scenes; temporally held-out portions of the SAME recordings and one path; no unseen-scene claim."
    write_csv(root / f"{prefix}_{split}_episodes.csv", rows)
    write_csv(root / f"{prefix}_{split}_summary.csv", summary)
    dump_json(root / f"{prefix}_{split}_summary.json", {
        "summary": summary, "split": split, "path_id": cfg.get("path_id"),
        "split_protocol": cfg.get("protocol", {"mode": "single_path"}),
        "checkpoint_epoch": checkpoint["epoch"], "bank_sha256": checkpoint["bank_sha256"],
        "evaluation_protocol": "reference-only observation; choose/freeze filter; score NEXT interval including secondary cold-start; reference FIR history retained; independent episodes",
        "reference_note": "scene_wiener is the unchanged per-sample prior; training-only or path-calibration-only FIR, not a global NR upper bound",
        "classification_scope": scope})
    for row in summary:
        if row["scene"] == "ALL" and row["path_id"] == "ALL":
            print(f"{row['method']}: mean NR={row['mean_primary_nr_db']:.3f} dB; max rebound={row['maximum_rebound_db']:.3f} dB")
    return summary
