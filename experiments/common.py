"""Provenance validation, losses, and identical evaluation episodes for baselines."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from anc.anc_simulator import simulate
from anc.metrics import score_signals
from data.dataset import ANCWindows
from data.inspect_data import inspect
from models.anc_transformer import ANCTransformer
from utils import bank_signature_cfg, device_for, dump_json, load_torch, sha256


def verified_bundle(cfg):
    bundle = load_torch(cfg["bank_file"])
    if bundle["format_version"] != 1 or bundle["config_signature"] != bank_signature_cfg(cfg):
        raise ValueError("Bank configuration changed. Rebuild bank for this config.")
    if bundle["manifest"] != inspect(cfg):
        raise ValueError("Dataset metadata/splits changed since bank creation. Rebuild bank.")
    if bundle["filters"].shape != (8, cfg["filter_length"]) or not torch.isfinite(bundle["filters"]).all():
        raise ValueError("Invalid Wiener bank shape/coefficients.")
    return bundle


def losses(result, labels, bank, filter_weight):
    target = bank[labels]
    relative_mse = ((result["filter"] - target).square().sum(-1) /
                    target.square().sum(-1).clamp_min(1e-12)).mean()
    classification = F.cross_entropy(result["logits"], labels)
    return {"loss": classification + filter_weight * relative_mse,
            "classification_loss": classification, "relative_filter_mse": relative_mse}


def loader_for(cfg, bundle, split, shuffle=False):
    dataset = ANCWindows(cfg, bundle["manifest"], split)
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["training"]["num_workers"], generator=generator)


def load_model(cfg, checkpoint_file=None):
    bundle = verified_bundle(cfg)
    checkpoint = load_torch(checkpoint_file or cfg["checkpoint_file"])
    if checkpoint["bank_sha256"] != sha256(cfg["bank_file"]):
        raise ValueError("Checkpoint belongs to a different bank. Retrain using this bank.")
    saved = checkpoint["config"]
    if saved["model"] != cfg["model"] or saved["dataset"]["control_seconds"] != cfg["dataset"]["control_seconds"]:
        raise ValueError("Model or future-control protocol differs from training checkpoint.")
    model = ANCTransformer(cfg, bundle)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if not torch.equal(model.filter_bank, bundle["filters"].float()):
        raise ValueError("Checkpoint has modified bank coefficients.")
    return model.to(device_for(cfg["training"]["device"])).eval(), bundle, checkpoint


@torch.no_grad()
def evaluate(cfg, model, bundle, split, methods=("transformer",)):
    allowed = {"transformer", "scene_wiener", "pooled_wiener", "mean_wiener"}
    if not methods or not set(methods).issubset(allowed):
        raise ValueError("Unknown/empty baseline list.")
    model.eval()
    device = next(model.parameters()).device
    bank = bundle["filters"].to(device)
    secondary = bundle["secondary_path"].to(device)
    pooled = bundle["pooled_filter"].to(device)
    rows = []
    for batch in loader_for(cfg, bundle, split):
        labels = batch["scene_id"].to(device)
        prediction = model(batch["observation"].to(device))
        reference = batch["reference"].to(device)
        disturbance = batch["disturbance"].to(device)
        context = int(batch["context"][0])
        candidates = {"transformer": prediction["filter"].double(),
                      "scene_wiener": bank[labels],
                      "pooled_wiener": pooled[None].expand(len(labels), -1),
                      "mean_wiener": bank.mean(0)[None].expand(len(labels), -1)}
        for method in methods:
            outcome = simulate(reference, disturbance, candidates[method], secondary, context)
            residual, control = outcome["residual"].cpu().numpy(), outcome["control"].cpu().numpy()
            target = disturbance[:, context:].cpu().numpy()
            for i in range(len(labels)):
                metrics = score_signals(target[i], residual[i], control[i], cfg["sample_rate"], **cfg["metrics"])
                row = {"method": method, "scene": batch["scene_name"][i],
                       "observation_start": int(batch["observation_start"][i]),
                       "control_start": int(batch["control_start"][i]), "scored_samples": len(target[i]),
                       "path_id": cfg["path_id"], **metrics}
                if method == "transformer":
                    row["correct_scene"] = int(prediction["logits"][i].argmax() == labels[i])
                    row["relative_filter_mse"] = float(((candidates[method][i] - bank[labels[i]]).square().sum() /
                                                        bank[labels[i]].square().sum().clamp_min(1e-12)).cpu())
                    for scene_id, name in enumerate(cfg["scene_names"]):
                        row[f"weight_{name}"] = float(prediction["weights"][i, scene_id].cpu())
                rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["scene"])].append(row)
        groups[(row["method"], "ALL")].append(row)
    summaries = []
    for (method, scene), values in sorted(groups.items()):
        item = {"method": method, "scene": scene, "episodes": len(values),
                "mean_primary_nr_db": float(np.mean([v["primary_nr_db"] for v in values])),
                "minimum_primary_nr_db": float(min(v["primary_nr_db"] for v in values)),
                "mean_broadband_nr_db": float(np.mean([v["broadband_nr_db"] for v in values])),
                "mean_episode_maximum_rebound_db": float(np.mean([v["maximum_rebound_db"] for v in values])),
                "maximum_rebound_db": float(max(v["maximum_rebound_db"] for v in values)),
                "mean_positive_band_rebound_db": float(np.mean([v["mean_positive_band_rebound_db"] for v in values])),
                "maximum_control_peak_abs": float(max(v["control_peak_abs"] for v in values))}
        if method == "transformer":
            item["scene_accuracy"] = float(np.mean([v["correct_scene"] for v in values]))
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
    write_csv(root / f"{prefix}_{split}_episodes.csv", rows)
    write_csv(root / f"{prefix}_{split}_summary.csv", summary)
    dump_json(root / f"{prefix}_{split}_summary.json", {
        "summary": summary, "split": split, "path_id": cfg["path_id"],
        "checkpoint_epoch": checkpoint["epoch"], "bank_sha256": checkpoint["bank_sha256"],
        "protocol": "reference-only observation; choose/freeze filter; score NEXT interval including secondary cold-start; reference FIR history retained; independent episodes",
        "reference_note": "scene_wiener uses the ground-truth scene label but training-only FIRs; it is not a global NR upper bound",
        "classification_scope": "8 known scenes, temporally held-out portions of the SAME recordings; no unseen-scene claim"})
    for row in summary:
        if row["scene"] == "ALL":
            print(f"{row['method']}: mean NR={row['mean_primary_nr_db']:.3f} dB; max rebound={row['maximum_rebound_db']:.3f} dB")
    return summary
