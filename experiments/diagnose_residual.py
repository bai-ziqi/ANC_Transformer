"""Locate where a prior-conditioned model loses performance versus its prior."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from anc.anc_simulator import simulate
from anc.metrics import score_signals, third_octave_band_power
from experiments.common import load_model, loader_for, write_csv
from utils import common_parser, dump_json, load_config


def _episode_summaries(rows):
    summaries = []
    for path_id in ("ALL", *sorted({int(row["path_id"]) for row in rows})):
        values = rows if path_id == "ALL" else [row for row in rows if int(row["path_id"]) == path_id]
        mse_deltas = np.asarray([row["residual_mse_delta"] for row in values], dtype=np.float64)
        nr_losses = np.asarray([row["primary_nr_delta_db"] for row in values], dtype=np.float64)
        positive_mse = np.maximum(mse_deltas, 0)
        positive_nr_loss = np.maximum(-nr_losses, 0)
        top_mse = np.sort(positive_mse)[-min(10, len(values)):].sum()
        top_nr = np.sort(positive_nr_loss)[-min(10, len(values)):].sum()
        summaries.append({
            "path_id": path_id,
            "episodes": len(values),
            "transformer_primary_nr_db": float(np.mean([row["transformer_primary_nr_db"] for row in values])),
            "prior_primary_nr_db": float(np.mean([row["prior_primary_nr_db"] for row in values])),
            "mean_primary_nr_delta_db": float(np.mean(nr_losses)),
            "transformer_residual_mse": float(np.mean([row["transformer_residual_mse"] for row in values])),
            "prior_residual_mse": float(np.mean([row["prior_residual_mse"] for row in values])),
            "mean_residual_mse_delta": float(np.mean(mse_deltas)),
            "percent_episodes_higher_mse": float(100 * np.mean(mse_deltas > 0)),
            "percent_episodes_lower_primary_nr": float(100 * np.mean(nr_losses < 0)),
            "mean_correction_relative_l2": float(np.mean([row["correction_relative_l2"] for row in values])),
            "p95_correction_relative_l2": float(np.percentile([row["correction_relative_l2"] for row in values], 95)),
            "positive_mse_delta_mass": float(positive_mse.sum()),
            "negative_mse_delta_mass": float(np.maximum(-mse_deltas, 0).sum()),
            "top10_positive_mse_share": float(top_mse / positive_mse.sum()) if positive_mse.sum() else 0.0,
            "top10_primary_nr_loss_share": float(top_nr / positive_nr_loss.sum()) if positive_nr_loss.sum() else 0.0,
        })
    return summaries


def _frequency_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[("ALL", row["centre_hz"])].append(row)
        grouped[(int(row["path_id"]), row["centre_hz"])].append(row)

    summaries = []
    for (path_id, centre), values in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        disturbance_power = sum(row["disturbance_band_power"] for row in values)
        transformer_power = sum(row["transformer_residual_band_power"] for row in values)
        prior_power = sum(row["prior_residual_band_power"] for row in values)
        pooled_transformer = 10 * np.log10(max(disturbance_power, 1e-30) / max(transformer_power, 1e-30))
        pooled_prior = 10 * np.log10(max(disturbance_power, 1e-30) / max(prior_power, 1e-30))
        summaries.append({
            "path_id": path_id,
            "centre_hz": centre,
            "episodes": len(values),
            "mean_transformer_nr_db": float(np.mean([row["transformer_nr_db"] for row in values])),
            "mean_prior_nr_db": float(np.mean([row["prior_nr_db"] for row in values])),
            "mean_episode_nr_delta_db": float(np.mean([row["nr_delta_db"] for row in values])),
            "pooled_transformer_nr_db": float(pooled_transformer),
            "pooled_prior_nr_db": float(pooled_prior),
            "pooled_nr_delta_db": float(pooled_transformer - pooled_prior),
        })
    return summaries


@torch.no_grad()
def diagnose(cfg, split="validation"):
    """Compare one model with its unchanged per-episode prior on paired data."""
    model, bundle, checkpoint = load_model(cfg)
    if bundle.get("format_version") != 2:
        raise ValueError("Residual diagnostics require a cross-path bank.")
    model.eval()
    device = next(model.parameters()).device
    episode_rows, band_rows = [], []

    for batch in loader_for(cfg, bundle, split):
        observation = batch["observation"].to(device)
        reference = batch["reference"].to(device)
        disturbance = batch["disturbance"].to(device)
        secondary = batch["secondary_path"].to(device)
        prior = batch["prior_filter"].to(device).double()
        prediction = model(observation, prior_filter=prior)["filter"].double()
        context = int(batch["context"][0])

        model_outcome = simulate(reference, disturbance, prediction, secondary, context)
        prior_outcome = simulate(reference, disturbance, prior, secondary, context)
        target_np = disturbance[:, context:].cpu().numpy()
        model_residual_np = model_outcome["residual"].cpu().numpy()
        prior_residual_np = prior_outcome["residual"].cpu().numpy()
        model_control_np = model_outcome["control"].cpu().numpy()
        prior_control_np = prior_outcome["control"].cpu().numpy()
        filter_denominator = prior.square().sum(-1).clamp_min(1e-12)
        relative_filter_mse = ((prediction - prior).square().sum(-1) / filter_denominator)

        for i in range(len(prior)):
            model_metrics = score_signals(
                target_np[i], model_residual_np[i], model_control_np[i],
                cfg["sample_rate"], **cfg["metrics"],
            )
            prior_metrics = score_signals(
                target_np[i], prior_residual_np[i], prior_control_np[i],
                cfg["sample_rate"], **cfg["metrics"],
            )
            model_mse = float(np.mean(model_residual_np[i] ** 2))
            prior_mse = float(np.mean(prior_residual_np[i] ** 2))
            residual_mse_delta = model_mse - prior_mse
            row = {
                "scene": batch["scene_name"][i],
                "scene_id": int(batch["scene_id"][i]),
                "path_id": int(batch["path_id"][i]),
                "observation_start": int(batch["observation_start"][i]),
                "control_start": int(batch["control_start"][i]),
                "transformer_primary_nr_db": model_metrics["primary_nr_db"],
                "prior_primary_nr_db": prior_metrics["primary_nr_db"],
                "primary_nr_delta_db": model_metrics["primary_nr_db"] - prior_metrics["primary_nr_db"],
                "transformer_broadband_nr_db": model_metrics["broadband_nr_db"],
                "prior_broadband_nr_db": prior_metrics["broadband_nr_db"],
                "broadband_nr_delta_db": model_metrics["broadband_nr_db"] - prior_metrics["broadband_nr_db"],
                "transformer_residual_mse": model_mse,
                "prior_residual_mse": prior_mse,
                "residual_mse_delta": residual_mse_delta,
                "residual_mse_relative_delta": residual_mse_delta / max(prior_mse, 1e-30),
                "transformer_maximum_rebound_db": model_metrics["maximum_rebound_db"],
                "prior_maximum_rebound_db": prior_metrics["maximum_rebound_db"],
                "maximum_rebound_delta_db": model_metrics["maximum_rebound_db"] - prior_metrics["maximum_rebound_db"],
                "relative_filter_mse": float(relative_filter_mse[i]),
                "correction_relative_l2": float(torch.sqrt(relative_filter_mse[i])),
            }
            episode_rows.append(row)

            centres, disturbance_bands = third_octave_band_power(
                target_np[i], cfg["sample_rate"], **cfg["metrics"],
            )
            model_centres, model_bands = third_octave_band_power(
                model_residual_np[i], cfg["sample_rate"], **cfg["metrics"],
            )
            prior_centres, prior_bands = third_octave_band_power(
                prior_residual_np[i], cfg["sample_rate"], **cfg["metrics"],
            )
            if not (np.array_equal(centres, model_centres) and np.array_equal(centres, prior_centres)):
                raise ValueError("Episode outputs produced inconsistent frequency centres.")
            for centre, d_band, model_band, prior_band in zip(
                centres, disturbance_bands, model_bands, prior_bands,
            ):
                if not 50 <= centre <= cfg["sample_rate"] / 2:
                    continue
                model_nr = 10 * np.log10(max(float(d_band), 1e-30) / max(float(model_band), 1e-30))
                prior_nr = 10 * np.log10(max(float(d_band), 1e-30) / max(float(prior_band), 1e-30))
                band_rows.append({
                    "scene": batch["scene_name"][i],
                    "path_id": int(batch["path_id"][i]),
                    "observation_start": int(batch["observation_start"][i]),
                    "centre_hz": float(centre),
                    "disturbance_band_power": float(d_band),
                    "transformer_residual_band_power": float(model_band),
                    "prior_residual_band_power": float(prior_band),
                    "transformer_nr_db": float(model_nr),
                    "prior_nr_db": float(prior_nr),
                    "nr_delta_db": float(model_nr - prior_nr),
                })

    episode_summary = _episode_summaries(episode_rows)
    frequency_summary = _frequency_summaries(band_rows)
    top_damage = sorted(episode_rows, key=lambda row: row["residual_mse_delta"], reverse=True)[:20]
    top_relative_damage = sorted(
        episode_rows, key=lambda row: row["residual_mse_relative_delta"], reverse=True,
    )[:20]
    top_nr_loss = sorted(episode_rows, key=lambda row: row["primary_nr_delta_db"])[:20]
    top_band_loss = [
        row for row in frequency_summary if row["path_id"] == "ALL"
    ]
    top_band_loss = sorted(top_band_loss, key=lambda row: row["mean_episode_nr_delta_db"])[:15]

    output_root = Path(cfg["output_dir"])
    write_csv(output_root / f"residual_diagnostic_{split}_episodes.csv", episode_rows)
    write_csv(output_root / f"residual_diagnostic_{split}_path_summary.csv", episode_summary)
    write_csv(output_root / f"residual_diagnostic_{split}_bands.csv", frequency_summary)
    write_csv(output_root / f"residual_diagnostic_{split}_top_damage.csv", top_damage)
    write_csv(output_root / f"residual_diagnostic_{split}_top_relative_damage.csv", top_relative_damage)
    write_csv(output_root / f"residual_diagnostic_{split}_top_nr_loss.csv", top_nr_loss)
    dump_json(output_root / f"residual_diagnostic_{split}.json", {
        "episode_summary": episode_summary,
        "top_band_loss": top_band_loss,
        "top_nr_loss": top_nr_loss,
        "top_relative_damage": top_relative_damage,
        "split": split,
        "checkpoint_epoch": checkpoint["epoch"],
        "filter_output_mode": cfg["model"].get("filter_output_mode"),
        "bank_sha256": checkpoint["bank_sha256"],
        "interpretation": (
            "All differences are model minus the unchanged per-episode Wiener prior. "
            "Positive residual_mse_delta and negative primary_nr_delta_db indicate degradation. "
            "Band summaries provide both equal-episode means and pooled-energy NR. "
            "The extended band table spans 50 Hz through Nyquist, while maximum_rebound_db "
            "retains the project's existing 1-8 kHz definition."
        ),
    })

    for row in episode_summary:
        print(
            f'path={row["path_id"]} n={row["episodes"]} '
            f'NR={row["transformer_primary_nr_db"]:.3f} dB '
            f'delta={row["mean_primary_nr_delta_db"]:+.3f} dB; '
            f'MSE delta={row["mean_residual_mse_delta"]:+.6e}; '
            f'correction L2={row["mean_correction_relative_l2"]:.4f}'
        )
    print("Top positive-MSE-delta episodes:")
    for row in top_damage[:10]:
        print(
            f'  {row["scene"]:<8s} path={row["path_id"]} start={row["observation_start"]} '
            f'NR delta={row["primary_nr_delta_db"]:+.3f} dB; '
            f'MSE delta={row["residual_mse_delta"]:+.6e}; '
            f'correction L2={row["correction_relative_l2"]:.4f}'
        )
    print("Top relative-MSE-delta episodes:")
    for row in top_relative_damage[:10]:
        print(
            f'  {row["scene"]:<8s} path={row["path_id"]} start={row["observation_start"]} '
            f'relative MSE delta={100 * row["residual_mse_relative_delta"]:+.2f}%; '
            f'NR delta={row["primary_nr_delta_db"]:+.3f} dB; '
            f'correction L2={row["correction_relative_l2"]:.4f}'
        )
    print("Largest primary-band NR losses:")
    for row in top_nr_loss[:10]:
        print(
            f'  {row["scene"]:<8s} path={row["path_id"]} start={row["observation_start"]} '
            f'NR delta={row["primary_nr_delta_db"]:+.3f} dB; '
            f'MSE delta={row["residual_mse_delta"]:+.6e}; '
            f'correction L2={row["correction_relative_l2"]:.4f}'
        )
    print("Largest average frequency-band losses (all paths):")
    for row in top_band_loss[:10]:
        print(
            f'  {row["centre_hz"]:7.1f} Hz: '
            f'delta={row["mean_episode_nr_delta_db"]:+.3f} dB'
        )
    print(f"Saved residual diagnostics under {output_root}")
    return {"path_summary": episode_summary, "frequency_summary": frequency_summary}


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()
    diagnose(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
