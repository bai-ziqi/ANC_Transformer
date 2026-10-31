"""Validate pairs and freeze chronological, sample-indexed split boundaries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from utils import common_parser, dump_json, load_config, sha256


def read_mono(path, start, frames):
    if start < 0 or frames <= 0:
        raise ValueError("Audio reads require start >= 0 and frames > 0.")
    value, _ = sf.read(str(path), start=int(start), frames=int(frames), dtype="float64", always_2d=True)
    if len(value) != frames:
        raise ValueError(f"Short/truncated read: {path}: wanted {frames}, got {len(value)}.")
    value = value.mean(axis=1)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Nonfinite audio samples: {path}.")
    return value


def load_secondary(cfg):
    paths = np.load(Path(cfg["data_root"]) / "sh.npy", allow_pickle=False)
    if paths.shape == (1967, 10):
        paths = paths.T
    if paths.shape != (10, 1967) or not np.all(np.isfinite(paths)):
        raise ValueError(f"Expected 10 x 1967 finite secondary paths; got {paths.shape}.")
    selected = np.asarray(paths[cfg["path_id"] - 1], dtype=np.float64)
    if np.dot(selected, selected) <= 0:
        raise ValueError("Selected secondary path is all zero.")
    return selected


def chronological_ranges(frames, cfg):
    sr, split = cfg["sample_rate"], cfg["split"]
    start = round(split["skip_seconds"] * sr)
    end = frames - round(split["tail_margin_seconds"] * sr)
    gap = round(split["gap_seconds"] * sr)
    usable = end - start - 2 * gap
    n_train = int(usable * split["fractions"][0])
    n_val = int(usable * split["fractions"][1])
    train = [start, start + n_train]
    val = [train[1] + gap, train[1] + gap + n_val]
    test = [val[1] + gap, end]
    cal_n = round(split["calibration_seconds"] * sr)
    cal_gap = round(split["calibration_gap_seconds"] * sr)
    calibration = [train[1] - cal_n, train[1]]
    fit = [train[0], calibration[0] - cal_gap]
    obs = round(cfg["features"]["observation_seconds"] * sr)
    score = round(cfg["dataset"]["control_seconds"] * sr)
    if usable <= 0 or fit[1] - fit[0] < 2 * (cfg["filter_length"] - 1) + cfg["filter_length"] + 1:
        raise ValueError("Recording is too short for Wiener fit and inner-train calibration.")
    if min(b - a for a, b in (train, val, test)) < obs + score:
        raise ValueError("Each outer split must fit observation + future control window.")
    return {"train": train, "validation": val, "test": test,
            "wiener_fit": fit, "wiener_calibration": calibration}


def inspect(cfg):
    root = Path(cfg["data_root"])
    secondary = load_secondary(cfg)
    scenes = []
    for name in cfg["scene_names"]:
        raw = root / "NOISE" / f"{name}.wav"
        expected = root / "EXPECTED_NOISE" / f"{name}_scene_{cfg['path_id']:02d}.wav"
        infos = [sf.info(str(p)) for p in (raw, expected)]
        if any(info.samplerate != cfg["sample_rate"] for info in infos):
            raise ValueError(f"Unexpected sample rate: {name}.")
        if infos[0].frames != infos[1].frames:
            raise ValueError(f"Reference/target length mismatch: {name}.")
        # Read-only metadata fingerprint; avoids repeatedly hashing >20 GB.
        sources = [{"path": str(p.resolve()), "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns, "frames": i.frames,
                    "sample_rate": i.samplerate, "channels": i.channels, "subtype": i.subtype}
                   for p, i in zip((raw, expected), infos)]
        scenes.append({"name": name, "sources": sources,
                       "frames": infos[0].frames,
                       "ranges": chronological_ranges(infos[0].frames, cfg)})
    return {"path_id": cfg["path_id"], "sample_rate": cfg["sample_rate"],
            "filter_length": len(secondary), "scene_names": cfg["scene_names"],
            "secondary_sha256": sha256(root / "sh.npy"), "scenes": scenes,
            "interval_convention": "[start, end), absolute sample indices; all reads stay within their split",
            "provenance_note": "WAV identity uses path/size/mtime/frames (not full content hash); sh.npy uses SHA-256."}


def main():
    args = common_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    manifest = inspect(cfg)
    destination = Path(cfg["output_dir"]) / "data_manifest.json"
    dump_json(destination, manifest)
    for scene in manifest["scenes"]:
        seconds = scene["frames"] / cfg["sample_rate"]
        print(f"{scene['name']}: {seconds:.3f} s; {scene['ranges']}")
    print(f"Validated 8 scene pairs and path {cfg['path_id']}. Manifest: {destination}")


if __name__ == "__main__":
    main()
