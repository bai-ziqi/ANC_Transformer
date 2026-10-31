"""Observe past x, freeze one filter, then score a separate future interval."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data.inspect_data import read_mono


class ANCWindows(Dataset):
    def __init__(self, cfg, manifest, split):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split.")
        self.observation = round(cfg["features"]["observation_seconds"] * cfg["sample_rate"])
        self.score = round(cfg["dataset"]["control_seconds"] * cfg["sample_rate"])
        # FIR reference history is observable at decision time. Secondary-path
        # state starts at zero; never synthesize anti-noise before the decision.
        self.context = cfg["filter_length"] - 1
        self.records = []
        count = cfg["dataset"][f"{split}_windows_per_scene"]
        rng = np.random.default_rng(cfg["seed"])
        for label, scene in enumerate(manifest["scenes"]):
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
            for start in starts:
                self.records.append({"scene_id": label, "scene_name": scene["name"],
                                     "observation_start": int(start), "split_start": a,
                                     "split_end": b, "raw": scene["sources"][0]["path"],
                                     "target": scene["sources"][1]["path"]})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        start = r["observation_start"]
        control_start = start + self.observation
        signal_start = control_start - self.context
        assert signal_start >= r["split_start"] and control_start + self.score <= r["split_end"]
        observation = read_mono(r["raw"], start, self.observation)
        reference = read_mono(r["raw"], signal_start, self.context + self.score)
        target = read_mono(r["target"], signal_start, self.context + self.score)
        return {"observation": torch.tensor(observation, dtype=torch.float32),
                "reference": torch.tensor(reference, dtype=torch.float64),
                "disturbance": torch.tensor(target, dtype=torch.float64),
                "scene_id": r["scene_id"], "scene_name": r["scene_name"],
                "observation_start": start, "control_start": control_start,
                "context": self.context}
