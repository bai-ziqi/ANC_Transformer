"""Scene-by-path episodes for path-disjoint prior-conditioned ANC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.inspect_data import chronological_ranges, load_secondary_paths, read_mono
from utils import sha256


def protocol_paths(cfg):
    protocol = cfg.get("protocol", {})
    if protocol.get("mode") != "cross_path":
        raise ValueError("Cross-path data requires protocol.mode=cross_path.")
    groups = {}
    for split in ("train", "validation", "test"):
        values = protocol.get(f"{split}_paths")
        if not values:
            raise ValueError(f"protocol.{split}_paths cannot be empty.")
        normalized = tuple(int(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"protocol.{split}_paths contains duplicates.")
        if any(not 1 <= value <= 10 for value in normalized):
            raise ValueError("Protocol paths must be integers in 1..10.")
        groups[split] = normalized
    names = ("train", "validation", "test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                raise ValueError(f"Path splits must be disjoint; shared: {sorted(overlap)}.")
    return groups


def _source_info(path):
    info = sf.info(str(path))
    return {
        "path": str(path.resolve()), "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns, "frames": info.frames,
        "sample_rate": info.samplerate, "channels": info.channels,
        "subtype": info.subtype,
    }


def inspect_cross_path(cfg):
    """Fingerprint every configured scene/path pair without reading samples."""
    groups = protocol_paths(cfg)
    path_ids = sorted(set().union(*[set(values) for values in groups.values()]))
    secondary = load_secondary_paths(cfg)
    root = Path(cfg["data_root"])
    scenes = []
    for name in cfg["scene_names"]:
        raw = root / "NOISE" / f"{name}.wav"
        raw_info = _source_info(raw)
        if raw_info["sample_rate"] != cfg["sample_rate"]:
            raise ValueError(f"Unexpected sample rate: {raw}")
        targets = {}
        for path_id in path_ids:
            target = root / "EXPECTED_NOISE" / f"{name}_scene_{path_id:02d}.wav"
            if not target.is_file():
                raise FileNotFoundError(f"Missing scene/path target: {target}")
            target_info = _source_info(target)
            if target_info["sample_rate"] != cfg["sample_rate"]:
                raise ValueError(f"Unexpected sample rate: {target}")
            if target_info["frames"] != raw_info["frames"]:
                raise ValueError(f"Reference/target length mismatch: {target}")
            targets[str(path_id)] = target_info
        scenes.append({
            "name": name, "frames": raw_info["frames"],
            "ranges": chronological_ranges(raw_info["frames"], cfg),
            "sources": {"raw": raw_info, "targets": targets},
        })
    return {
        "protocol_mode": "cross_path", "path_ids": path_ids,
        "path_splits": {key: list(value) for key, value in groups.items()},
        "sample_rate": cfg["sample_rate"], "filter_length": len(secondary[0]),
        "scene_names": cfg["scene_names"], "secondary_sha256": sha256(root / "sh.npy"),
        "scenes": scenes,
        "interval_convention": "[start, end), absolute sample indices; reads stay within split",
    }


class CrossPathWindows(Dataset):
    """Episodes paired across all paths in one split.

    Window locations are shared across paths so comparisons use identical
    reference ranges. Each item carries its own Wiener prior and plant.
    """

    def __init__(self, cfg, manifest, split, priors, secondary_paths):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split.")
        self.paths = protocol_paths(cfg)[split]
        self.path_ids = tuple(int(value) for value in manifest["path_ids"])
        self.bank_indices = tuple(self.path_ids.index(path_id) for path_id in self.paths)
        self.priors = priors
        self.secondary_paths = secondary_paths
        self.observation = round(cfg["features"]["observation_seconds"] * cfg["sample_rate"])
        self.score = round(cfg["dataset"]["control_seconds"] * cfg["sample_rate"])
        self.context = cfg["filter_length"] - 1
        self.records = []
        count = cfg["dataset"][f"{split}_windows_per_scene"]
        rng = np.random.default_rng(cfg["seed"])
        for scene_id, scene in enumerate(manifest["scenes"]):
            a, b = scene["ranges"][split]
            latest = b - self.observation - self.score
            if latest < a:
                raise ValueError(f"{scene['name']}: {split} is too short.")
            if split == "train":
                starts = rng.integers(a, latest + 1, size=count)
            else:
                starts = np.linspace(a, latest, count, dtype=np.int64)
                if len(starts) > 1 and np.min(np.diff(starts)) < self.observation + self.score:
                    raise ValueError(f"Reduce {split}_windows_per_scene: evaluation episodes would overlap.")
            for path_id, bank_index in zip(self.paths, self.bank_indices):
                for start in starts:
                    self.records.append({
                        "scene_id": scene_id, "scene_name": scene["name"],
                        "path_id": path_id, "bank_index": bank_index,
                        "observation_start": int(start), "split_start": a, "split_end": b,
                        "raw": scene["sources"]["raw"]["path"],
                        "target": scene["sources"]["targets"][str(path_id)]["path"],
                    })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        start = record["observation_start"]
        control_start = start + self.observation
        signal_start = control_start - self.context
        if signal_start < record["split_start"] or control_start + self.score > record["split_end"]:
            raise ValueError("Episode crosses its declared split boundary.")
        observation = read_mono(record["raw"], start, self.observation)
        reference = read_mono(record["raw"], signal_start, self.context + self.score)
        target = read_mono(record["target"], signal_start, self.context + self.score)
        return {
            "observation": torch.tensor(observation, dtype=torch.float32),
            "reference": torch.tensor(reference, dtype=torch.float64),
            "disturbance": torch.tensor(target, dtype=torch.float64),
            "prior_filter": self.priors[record["scene_id"], record["bank_index"]].float(),
            "secondary_path": self.secondary_paths[record["bank_index"]].double(),
            "scene_id": record["scene_id"], "path_id": record["path_id"],
            "scene_name": record["scene_name"],
            "observation_start": start, "control_start": control_start,
            "context": self.context,
        }
