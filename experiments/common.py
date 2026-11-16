"""Provenance validation, losses, and identical offline clips for baselines."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anc.anc_simulator import simulate
from anc.metrics import score_signals
from data.sfanc import SfancDataset, load_training_bundle
from models.anc_transformer import ANCTransformer
from utils import device_for, dump_json, load_torch, sha256


def verified_bundle(cfg):
    return load_training_bundle(cfg)


def acoustic_mse(result, batch, secondary=None):
    """Differentiable residual MSE over the configured full-clip interval."""
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
    target = batch.get("target_filter")
    if target is None:
        raise ValueError("SFANC supervision requires target_filter; prior_filter is an input, not a label.")
    if not isinstance(target, torch.Tensor) or target.shape != result["filter"].shape:
        raise ValueError("target_filter must have exactly the predicted filter shape [B,filter_length].")
    target = target.to(result["filter"].device, dtype=result["filter"].dtype)
    relative_mse = ((result["filter"] - target).square().sum(-1) /
                    target.square().sum(-1).clamp_min(1e-12)).mean()
    mode = cfg["training"]["loss_mode"]
    filter_weight = cfg["training"]["filter_loss_weight"]
    acoustic_weight = cfg["training"]["acoustic_loss_weight"]
    values = {"relative_filter_mse": relative_mse}
    if mode in ("acoustic", "combined"):
        acoustic = acoustic_mse(result, batch, secondary)
        values["acoustic_mse"] = acoustic
    if mode in ("filter", "combined"):
        values["weighted_filter_loss"] = filter_weight * relative_mse
    if mode in ("acoustic", "combined"):
        values["weighted_acoustic_loss"] = acoustic_weight * acoustic
    values["loss"] = sum(value for key, value in values.items() if key.endswith("_loss"))
    return values


def prior_for(batch):
    """Return the independently calibrated prior for each sample's environment."""
    return batch["prior_filter"]


def loader_for(cfg, bundle, split, shuffle=False):
    dataset = SfancDataset(cfg, bundle, split)
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["training"]["num_workers"], generator=generator)


def load_model(cfg, checkpoint_file=None):
    bundle = verified_bundle(cfg)
    checkpoint = load_torch(checkpoint_file or cfg["checkpoint_file"])
    if checkpoint.get("format_version") != 5:
        raise ValueError("Checkpoint is incompatible with the SFANC experiment. Retrain it.")
    if checkpoint["bank_sha256"] != sha256(cfg["bank_file"]):
        raise ValueError("Checkpoint belongs to a different bank. Retrain using this bank.")
    saved = checkpoint["config"]
    if checkpoint.get("targets_index_sha256") != sha256(Path(cfg["targets_dir"]) / "index.json"):
        raise ValueError("Checkpoint belongs to a different target-label index. Retrain using these labels.")
    keys = ("model", "features", "dataset", "protocol", "sample_rate", "filter_length")
    if any(saved.get(key) != cfg.get(key) for key in keys):
        raise ValueError("SFANC model, features, lengths or clip protocol differs from training checkpoint.")
    for key in ("loss_mode", "acoustic_loss_weight", "filter_loss_weight"):
        if saved["training"].get(key) != cfg["training"].get(key):
            raise ValueError(f"Training objective differs at {key}. Retrain the checkpoint.")
    model = ANCTransformer(cfg, bundle)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device_for(cfg["training"]["device"])).eval(), bundle, checkpoint


@torch.no_grad()
def evaluate(cfg, model, bundle, split, methods=("transformer",)):
    allowed = {"transformer", "zero_filter", "environment_prior", "sample_wiener"}
    if not methods or not set(methods).issubset(allowed):
        raise ValueError("Unknown/empty SFANC baseline list.")
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for batch in loader_for(cfg, bundle, split):
        reference = batch["reference"].to(device, dtype=torch.float64)
        disturbance = batch["disturbance"].to(device, dtype=torch.float64)
        secondary = batch["secondary_path"].to(device, dtype=torch.float64)
        prior = batch["prior_filter"].to(device, dtype=torch.float64)
        target = batch["target_filter"].to(device, dtype=torch.float64)
        prediction = model(batch["observation"].to(device), prior_filter=prior)
        candidates = {
            "transformer": prediction["filter"].double(),
            "zero_filter": torch.zeros_like(prior),
            "environment_prior": prior,
            "sample_wiener": target,
        }
        context = int(batch["context"][0])
        for method in methods:
            coefficients = candidates[method]
            outcome = simulate(reference, disturbance, coefficients, secondary, context)
            residual = outcome["residual"].cpu().numpy()
            control = outcome["control"].cpu().numpy()
            initial = disturbance[:, context:].cpu().numpy()
            for i in range(len(initial)):
                metrics = score_signals(initial[i], residual[i], control[i],
                                        bundle["sample_rate"], **cfg["metrics"])
                error = (coefficients[i] - target[i]).square().sum()
                error = error / target[i].square().sum().clamp_min(1e-12)
                change = torch.linalg.vector_norm(coefficients[i] - prior[i])
                change = change / torch.linalg.vector_norm(prior[i]).clamp_min(1e-12)
                rows.append({
                    "method": method,
                    "sample_id": str(batch["sample_id"][i]),
                    "environment_id": str(batch["environment_id"][i]),
                    "scored_samples": len(initial[i]),
                    "residual_mse": float(np.mean(residual[i] ** 2)),
                    "disturbance_mse": float(np.mean(initial[i] ** 2)),
                    "relative_filter_mse": float(error.cpu()),
                    "relative_prior_change_l2": float(change.cpu()),
                    **metrics,
                })
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for environment in (row["environment_id"], "ALL"):
            groups[(row["method"], environment)].append(row)
    summaries = []
    for (method, environment), values in sorted(groups.items()):
        summaries.append({
            "method": method, "environment_id": environment, "episodes": len(values),
            "mean_primary_nr_db": float(np.mean([v["primary_nr_db"] for v in values])),
            "minimum_primary_nr_db": float(min(v["primary_nr_db"] for v in values)),
            "mean_broadband_nr_db": float(np.mean([v["broadband_nr_db"] for v in values])),
            "mean_residual_mse": float(np.mean([v["residual_mse"] for v in values])),
            "relative_filter_mse": float(np.mean([v["relative_filter_mse"] for v in values])),
            "mean_relative_prior_change_l2": float(np.mean([v["relative_prior_change_l2"] for v in values])),
            "maximum_rebound_db": float(max(v["maximum_rebound_db"] for v in values)),
            "mean_episode_maximum_rebound_db": float(np.mean([v["maximum_rebound_db"] for v in values])),
            "mean_positive_band_rebound_db": float(np.mean([v["mean_positive_band_rebound_db"] for v in values])),
            "maximum_control_peak_abs": float(max(v["control_peak_abs"] for v in values)),
        })
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
    scope = ("SFANC clip-wise supervised FIR experiment. Unseen-path claims require disjoint "
             "configured environments; one shared path measures only unseen-noise adaptation.")
    write_csv(root / f"{prefix}_{split}_episodes.csv", rows)
    write_csv(root / f"{prefix}_{split}_summary.csv", summary)
    evaluation_protocol = (
        "Offline clip-wise evaluation; filter generated from configured observation and independent "
        "environment prior, then frozen and scored on the whole clip with zero initial state. "
        "Same boundary as sample-label selection; this is not an online next-window experiment."
    )
    reference_note = (
        "sample_wiener is the cached verified pure finite-window least-squares reference "
        "fitted to this clip; "
        "environment_prior is independently calibrated and is the only filter input. "
        "Primary NR is an auxiliary 50-5000 Hz equal-band statistic; label selection uses residual MSE."
    )
    dump_json(root / f"{prefix}_{split}_summary.json", {
        "summary": summary, "split": split,
        "filter_output_mode": cfg.get("model", {}).get("filter_output_mode"),
        "split_protocol": cfg["protocol"],
        "checkpoint_epoch": checkpoint["epoch"], "bank_sha256": checkpoint["bank_sha256"],
        "targets_index_sha256": checkpoint["targets_index_sha256"],
        "evaluation_protocol": evaluation_protocol,
        "reference_note": reference_note,
        "evaluation_scope": scope})
    for row in summary:
        if row["environment_id"] == "ALL":
            print(f"{row['method']}: residual MSE={row['mean_residual_mse']:.6e}; "
                  f"broadband NR={row['mean_broadband_nr_db']:.3f} dB")
    return summary
