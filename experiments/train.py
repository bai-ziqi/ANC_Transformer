"""Train CE + relative FIR error. Selection uses outer validation loss ONLY."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.common import evaluate, loader_for, losses, summarize, verified_bundle, write_csv
from models.anc_transformer import ANCTransformer
from utils import common_parser, device_for, dump_json, load_config, seed_everything, sha256


def train(cfg):
    seed_everything(cfg["seed"])
    device = device_for(cfg["training"]["device"])
    bundle = verified_bundle(cfg)
    model = ANCTransformer(cfg, bundle).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    train_loader = loader_for(cfg, bundle, "train", shuffle=True)
    validation_loader = loader_for(cfg, bundle, "validation")
    bank = model.filter_bank
    history, best = [], float("inf")
    destination = Path(cfg["checkpoint_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    bank_digest = sha256(cfg["bank_file"])
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device={device}; trainable parameters={parameters}; train episodes={len(train_loader.dataset)}", flush=True)
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        row = {"epoch": epoch}
        for split, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(split == "train")
            totals = {"loss": 0.0, "classification_loss": 0.0, "relative_filter_mse": 0.0}
            seen, correct = 0, 0
            with torch.set_grad_enabled(split == "train"):
                for batch in loader:
                    labels = batch["scene_id"].to(device)
                    prediction = model(batch["observation"].to(device))
                    objective = losses(prediction, labels, bank, cfg["training"]["filter_loss_weight"])
                    if not torch.isfinite(objective["loss"]):
                        raise ValueError("Nonfinite training/validation loss.")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        objective["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                        optimizer.step()
                    size = len(labels)
                    seen += size
                    correct += int((prediction["logits"].argmax(-1) == labels).sum())
                    for key, value in objective.items():
                        totals[key] += float(value.detach()) * size
            for key, value in totals.items():
                row[f"{split}_{key}"] = value / seen
            row[f"{split}_accuracy"] = correct / seen
        history.append(row)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save({"format_version": 1, "epoch": epoch, "validation_loss": best,
                        "selection_metric": "minimum outer validation CE + relative FIR MSE",
                        "config": cfg, "bank_sha256": bank_digest,
                        "model_state": model.state_dict(), "trainable_parameters": parameters}, destination)
        dump_json(Path(cfg["output_dir"]) / "training_history.json", history)
        print(f"epoch={epoch:03d} train={row['train_loss']:.4f} val={row['validation_loss']:.4f} "
              f"val_accuracy={row['validation_accuracy']:.3f}", flush=True)
    # Report acoustic validation for the selected checkpoint; TEST is untouched.
    from utils import load_torch
    selected = load_torch(destination)
    model.load_state_dict(selected["model_state"])
    validation_rows = evaluate(cfg, model, bundle, "validation")
    write_csv(Path(cfg["output_dir"]) / "selected_validation_episodes.csv", validation_rows)
    dump_json(Path(cfg["output_dir"]) / "selected_validation_summary.json", summarize(validation_rows))
    print(f"Selected epoch {selected['epoch']}; checkpoint: {destination}", flush=True)
    return selected


def main():
    args = common_parser(__doc__).parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
