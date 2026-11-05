"""Optimize acoustic residual energy; select on outer validation loss only."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.common import evaluate, loader_for, losses, prior_for, save_evaluation, verified_bundle
from models.anc_transformer import ANCTransformer
from utils import common_parser, device_for, dump_json, load_config, seed_everything, sha256


def _run_pass(cfg, model, loader, device, prior_bank, optimizer=None):
    """Run one training or inference pass and return size-weighted losses."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "acoustic_mse": 0.0, "relative_filter_mse": 0.0}
    seen, correct = 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch["scene_id"].to(device)
            prior_filter = prior_for(model, prior_bank, labels, batch)
            prediction = model(batch["observation"].to(device), prior_filter=prior_filter)
            objective = losses(prediction, cfg, batch)
            if not torch.isfinite(objective["loss"]):
                raise ValueError("Nonfinite training/validation loss.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                optimizer.step()
            size = len(labels)
            seen += size
            if "logits" in prediction:
                correct += int((prediction["logits"].argmax(-1) == labels).sum())
            for key, value in objective.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
    result = {key: value / seen for key, value in totals.items()}
    if "logits" in prediction:
        result["accuracy"] = correct / seen
    return result


def train(cfg):
    seed_everything(cfg["seed"])
    device = device_for(cfg["training"]["device"])
    bundle = verified_bundle(cfg)
    model = ANCTransformer(cfg, bundle).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    train_loader = loader_for(cfg, bundle, "train", shuffle=True)
    validation_loader = loader_for(cfg, bundle, "validation")
    prior_bank = bundle["filters"].to(device).float()
    history, best = [], float("inf")
    best_epoch = None
    destination = Path(cfg["checkpoint_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    bank_digest = sha256(cfg["bank_file"])
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device={device}; trainable parameters={parameters}; train episodes={len(train_loader.dataset)}", flush=True)

    def save_checkpoint(epoch, validation_loss):
        metric_name = (
            f"minimum outer validation {cfg['training']['loss_mode']} loss; "
            "epoch 0 untrained baseline included"
        )
        torch.save({
            "format_version": 4,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "selection_metric": metric_name,
            "selection_policy": "strictly lower validation loss; epoch 0 baseline is eligible",
            "config": cfg,
            "bank_sha256": bank_digest,
            "model_state": model.state_dict(),
            "trainable_parameters": parameters,
        }, destination)

    # Residual mode starts exactly at w_prior. Evaluate that state before any
    # optimizer step so training can never erase a useful prior without proof
    # of lower outer-validation loss. Direct mode also gets the same baseline.
    baseline = _run_pass(cfg, model, validation_loader, device, prior_bank)
    baseline_row = {
        "epoch": 0,
        "baseline": True,
        "optimizer_steps": 0,
        **{f"validation_{key}": value for key, value in baseline.items()},
    }
    history.append(baseline_row)
    best = baseline_row["validation_loss"]
    best_epoch = 0
    save_checkpoint(0, best)
    dump_json(Path(cfg["output_dir"]) / "training_history.json", history)
    extra = (f" val_accuracy={baseline_row['validation_accuracy']:.3f}"
             if "validation_accuracy" in baseline_row else "")
    print(f"epoch=000 baseline val={best:.6e}{extra}", flush=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        row = {"epoch": epoch, "baseline": False}
        train_values = _run_pass(cfg, model, train_loader, device, prior_bank, optimizer)
        validation_values = _run_pass(cfg, model, validation_loader, device, prior_bank)
        row.update({f"train_{key}": value for key, value in train_values.items()})
        row.update({f"validation_{key}": value for key, value in validation_values.items()})
        history.append(row)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            best_epoch = epoch
            save_checkpoint(epoch, best)
        dump_json(Path(cfg["output_dir"]) / "training_history.json", history)
        extra = (f" val_accuracy={row['validation_accuracy']:.3f}"
                 if "validation_accuracy" in row else "")
        print(f"epoch={epoch:03d} train={row['train_loss']:.6e} val={row['validation_loss']:.6e}{extra}", flush=True)
    # Report acoustic validation for the selected checkpoint; TEST is untouched.
    from utils import load_torch
    selected = load_torch(destination)
    if selected["epoch"] != best_epoch:
        raise RuntimeError("Selected checkpoint does not match the validation-selection state.")
    model.load_state_dict(selected["model_state"])
    validation_rows = evaluate(cfg, model, bundle, "validation")
    save_evaluation(cfg, validation_rows, "validation", "selected", selected)
    print(f"Selected epoch {selected['epoch']}; checkpoint: {destination}", flush=True)
    return selected


def main():
    args = common_parser(__doc__).parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
