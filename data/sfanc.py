"""Clip-level SFANC data, environment pairing, and train-only feature statistics."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from anc.anc_simulator import causal_numpy
from data.environments import load_environments, read_vector, resolve_sample_rate
from data.preprocess import LogSpectrum
from utils import bank_signature_cfg, dump_json, load_config, load_torch, sha256


_SPLITS = ("train", "validation", "test")


def _observation_samples(cfg, frames):
    seconds = cfg["features"].get("observation_seconds")
    samples = frames if seconds is None else round(seconds * cfg["sample_rate"])
    if not cfg["features"]["n_fft"] <= samples <= frames:
        raise ValueError("Observation must fit the clip and cover the feature FFT.")
    return samples


def inspect_sfanc(cfg, limit=None, *, save=True):
    """Inspect original clip splits and create deterministic clip/environment pairs."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive number of source clips per split.")
    root = Path(cfg["data_root"])
    settings = cfg["sfanc"]
    environments = load_environments(cfg)
    environment_ids = list(environments)
    extensions = {str(value).lower() for value in settings.get("extensions", [".wav"])}
    pairing = settings.get("pairing", "round_robin")
    if pairing not in ("round_robin", "all"):
        raise ValueError("sfanc.pairing must be round_robin or all.")
    records, source_counts, seen_paths, frame_counts = [], {}, {}, set()
    split_roots = []
    for split in _SPLITS:
        directory = (root / settings["split_dirs"][split]).resolve()
        if any(directory == other or directory in other.parents or other in directory.parents
               for other in split_roots):
            raise ValueError("SFANC split directories must be distinct and must not contain each other.")
        split_roots.append(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {directory}.")
        paths = sorted((path for path in directory.rglob("*") if path.is_file()
                        and path.suffix.lower() in extensions), key=lambda path: path.as_posix())
        source_counts[split] = len(paths)
        expected = settings.get("expected_counts", {}).get(split)
        if expected is not None and len(paths) != expected:
            raise ValueError(f"{split}: expected {expected} source clips, found {len(paths)} in {directory}.")
        if not paths:
            raise ValueError(f"No supported noise clips in {directory}.")
        active_ids = [str(value) for value in cfg["protocol"][f"{split}_environments"]]
        if not active_ids or len(set(active_ids)) != len(active_ids) or not set(active_ids) <= set(environment_ids):
            raise ValueError(f"Invalid {split} environment ids: {active_ids}.")
        for index, path in enumerate(paths):
            resolved = path.resolve()
            if resolved in seen_paths:
                raise ValueError(f"A source file is shared between splits: {resolved}.")
            seen_paths[resolved] = split
            if limit is not None and index >= limit:
                continue
            if path.suffix.lower() in (".wav", ".flac"):
                info = sf.info(path)
                if info.channels != 1:
                    raise ValueError(f"Expected mono clip, got {info.channels} channels: {path}.")
                frames, rate = info.frames, info.samplerate
            else:
                value, metadata = read_vector(path, settings.get("waveform_key"))
                frames, rate = len(value), metadata["sample_rate"]
            rate = resolve_sample_rate(cfg.get("sample_rate"), rate, str(path))
            if rate != cfg["sample_rate"]:
                raise ValueError(f"Unexpected clip sample rate: {path}.")
            if frames <= cfg["filter_length"]:
                raise ValueError(f"Clip must exceed the control FIR length: {path}.")
            _observation_samples(cfg, frames)
            frame_counts.add(frames)
            stat = path.stat()
            sample_id = f"{split}/{path.relative_to(directory).as_posix()}"
            selected = active_ids if pairing == "all" else [active_ids[index % len(active_ids)]]
            for environment_id in selected:
                record_id = hashlib.sha256(f"{sample_id}\0{environment_id}".encode("utf-8")).hexdigest()
                records.append({
                    "record_id": record_id, "sample_id": sample_id, "split": split,
                    "environment_id": environment_id, "path": str(resolved),
                    "frames": frames, "sample_rate": rate,
                    "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                })
    if len(frame_counts) != 1:
        raise ValueError("SFANC clips must have one common length across splits; no implicit crop/pad is applied.")
    manifest = {
        "format_version": 3, "protocol_mode": "sfanc", "sample_rate": cfg["sample_rate"],
        "filter_length": cfg["filter_length"], "primary_length": cfg["primary_length"],
        "secondary_length": cfg["secondary_length"], "environment_ids": environment_ids,
        "environments": {key: {"sample_rate": value["sample_rate"], **value["provenance"]}
                         for key, value in environments.items()},
        "protocol": cfg["protocol"], "pairing": pairing, "records": records,
        "source_counts": source_counts, "expected_counts": settings.get("expected_counts", {}),
        "frames": next(iter(frame_counts)), "partial": limit is not None,
        "limit_per_split": limit,
        "evaluation_protocol": "offline same-clip observation/label/evaluation; causal zero-state filtering; full clip scored",
        "identity_note": "Split membership is preserved. Resolved source-path duplicates are rejected; copied content and overlapping source recordings are not inferred from filenames.",
    }
    if save:
        dump_json(cfg["manifest_file"], manifest)
    return manifest


def load_noise(record, cfg):
    """Load exactly the inspected mono clip; never resample, crop, or normalize it."""
    path = Path(record["path"])
    stat = path.stat()
    if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
        raise ValueError(f"Source clip changed since inspection: {path}.")
    value, metadata = read_vector(path, cfg["sfanc"].get("waveform_key"))
    rate = resolve_sample_rate(cfg["sample_rate"], metadata["sample_rate"], str(path))
    if rate != record["sample_rate"] or len(value) != record["frames"]:
        raise ValueError(f"Clip dimensions changed since inspection: {path}.")
    return value


def fit_sfanc_normalizer(cfg, manifest):
    """Fit log-spectrum normalization on TRAIN noise only, deduplicated by clip."""
    unique = {record["sample_id"]: record for record in manifest["records"] if record["split"] == "train"}
    records = list(unique.values())
    maximum = int(cfg["features"].get("normalization_max_clips", 256))
    if maximum <= 0 or not records:
        raise ValueError("Feature normalization requires training clips and a positive normalization_max_clips.")
    indices = np.linspace(0, len(records) - 1, min(maximum, len(records)), dtype=np.int64)
    transform = LogSpectrum(cfg["features"])
    total, squared, frame_count, sample_ids = None, None, 0, []
    with torch.no_grad():
        for index in indices:
            record = records[int(index)]
            value = load_noise(record, cfg)
            observation = value[:_observation_samples(cfg, len(value))]
            features = transform(torch.tensor(observation[None], dtype=torch.float32)).squeeze(0).double()
            first, second = features.sum(0), features.square().sum(0)
            total = first if total is None else total + first
            squared = second if squared is None else squared + second
            frame_count += len(features)
            sample_ids.append(record["sample_id"])
    mean = total / frame_count
    std = (squared / frame_count - mean.square()).clamp_min(1e-4).sqrt()
    return {"mean": mean.float(), "std": std.float(), "frames": frame_count,
            "source_split": "train", "sample_ids": sample_ids}


def _validate_manifest_sources(cfg, manifest):
    """Check clip membership and stat fingerprints without decoding all waveforms."""
    settings = cfg["sfanc"]
    extensions = {str(value).lower() for value in settings.get("extensions", [".wav"])}
    for split in _SPLITS:
        directory = (Path(cfg["data_root"]) / settings["split_dirs"][split]).resolve()
        paths = sorted((path for path in directory.rglob("*") if path.is_file()
                        and path.suffix.lower() in extensions), key=lambda path: path.as_posix())
        if len(paths) != manifest["source_counts"][split]:
            raise ValueError(f"SFANC {split} source count changed; rebuild manifest and targets.")
        limit = manifest.get("limit_per_split")
        if limit is not None:
            paths = paths[:limit]
        records = {record["path"]: record for record in manifest["records"] if record["split"] == split}
        if {str(path.resolve()) for path in paths} != set(records):
            raise ValueError(f"SFANC {split} source membership changed; rebuild manifest and targets.")
        for path, record in records.items():
            stat = Path(path).stat()
            if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise ValueError(f"Source clip changed since target generation: {path}.")


def load_training_bundle(cfg):
    """Validate the independent prior bank and complete, manifest-bound targets."""
    manifest_path = Path(cfg["manifest_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 3 or manifest.get("protocol_mode") != "sfanc":
        raise ValueError("Expected an SFANC format-3 manifest; rebuild the data manifest.")
    if manifest.get("partial") and not cfg.get("training", {}).get("allow_partial", False):
        raise ValueError("Manifest is partial. Rebuild all targets, or explicitly set training.allow_partial=true for a smoke run.")
    for name in ("sample_rate", "filter_length", "primary_length", "secondary_length"):
        if manifest.get(name) != cfg[name]:
            raise ValueError(f"SFANC manifest differs from config at {name}; rebuild targets.")
    if manifest.get("protocol") != cfg["protocol"] or manifest.get("pairing") != cfg["sfanc"].get("pairing", "round_robin"):
        raise ValueError("SFANC split/pairing protocol changed; rebuild targets.")
    _validate_manifest_sources(cfg, manifest)
    bundle = load_torch(cfg["bank_file"])
    if bundle.get("format_version") != 3 or bundle.get("config_signature") != bank_signature_cfg(cfg):
        raise ValueError("Environment prior configuration changed; rebuild priors.")
    if bundle.get("sample_rate") != cfg["sample_rate"] or bundle.get("filter_length") != cfg["filter_length"]:
        raise ValueError("Environment prior sample rate or control FIR length differs from config.")
    environment_ids = list(bundle.get("environment_ids", []))
    if environment_ids != manifest.get("environment_ids"):
        raise ValueError("Prior bank and sample manifest environment ids differ.")
    environments = load_environments(cfg)
    current_provenance = {key: {"sample_rate": value["sample_rate"], **value["provenance"]}
                          for key, value in environments.items()}
    if manifest.get("environments") != current_provenance:
        raise ValueError("Acoustic paths changed since inspection; rebuild manifest, priors, and targets.")
    shapes = {"prior_filters": (len(environment_ids), cfg["filter_length"]),
              "primary_paths": (len(environment_ids), cfg["primary_length"]),
              "secondary_paths": (len(environment_ids), cfg["secondary_length"])}
    for key, shape in shapes.items():
        value = bundle.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid environment bank {key}; expected finite {shape}.")
    for index, environment_id in enumerate(environment_ids):
        environment = environments[environment_id]
        for key, source in (("primary_paths", "primary"), ("secondary_paths", "secondary")):
            actual = bundle[key][index].double().numpy()
            if not np.array_equal(actual, environment[source]):
                raise ValueError(f"Prior bank has different {source} coefficients for {environment_id}.")
    target_root = Path(cfg["targets_dir"])
    index_path = target_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format_version") != 1 or index.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Target cache belongs to another manifest; regenerate targets.")
    if index.get("config_signature") != bank_signature_cfg(cfg) or index.get("filter_length") != cfg["filter_length"]:
        raise ValueError("Target cache configuration changed; regenerate targets.")
    if index.get("selection_metric") != "minimum_causal_residual_mse":
        raise ValueError("Target cache does not use the agreed minimum residual-MSE selection.")
    records = manifest["records"]
    if not index.get("complete") or index.get("completed_records") != len(records) or index.get("total_records") != len(records):
        raise ValueError("Target cache is incomplete; resume label generation before training.")
    covered = 0
    for chunk in index.get("chunks", []):
        if chunk.get("start") != covered or not isinstance(chunk.get("count"), int) or chunk["count"] <= 0:
            raise ValueError("Target chunks must cover manifest records in contiguous order.")
        file = (target_root / chunk["file"]).resolve()
        if target_root.resolve() not in file.parents:
            raise ValueError("Target chunk path escapes targets_dir.")
        if sha256(file) != chunk["sha256"]:
            raise ValueError(f"Target chunk checksum mismatch: {file}.")
        covered += chunk["count"]
    if covered != len(records):
        raise ValueError("Target chunk coverage differs from manifest record count.")
    bundle.update({"manifest": manifest, "targets_index": index, "targets_dir": str(target_root),
                   "targets_index_sha256": sha256(index_path), "filters": bundle["prior_filters"],
                   "normalizer": fit_sfanc_normalizer(cfg, manifest)})
    return bundle


class SfancDataset(Dataset):
    """Return separate environment priors and sample targets, scored on the full clip."""

    def __init__(self, cfg, bundle, split):
        if split not in _SPLITS:
            raise ValueError(f"Unknown SFANC split: {split}.")
        self.cfg, self.bundle = cfg, bundle
        self.records = [(index, record) for index, record in enumerate(bundle["manifest"]["records"])
                        if record["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} records in the target manifest.")
        self.environment_indices = {value: index for index, value in enumerate(bundle["environment_ids"])}
        self.chunks = bundle["targets_index"]["chunks"]
        self.chunk_ends = np.asarray([chunk["start"] + chunk["count"] for chunk in self.chunks])
        self.cache_limit = cfg["sfanc"].get("target_cache_chunks")
        if self.cache_limit is not None and (isinstance(self.cache_limit, bool)
                                             or not isinstance(self.cache_limit, int) or self.cache_limit <= 0):
            raise ValueError("sfanc.target_cache_chunks must be null or a positive integer.")
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _target(self, global_index, record):
        chunk_index = int(np.searchsorted(self.chunk_ends, global_index, side="right"))
        chunk = self.chunks[chunk_index]
        if chunk_index not in self.cache:
            with np.load(Path(self.bundle["targets_dir"]) / chunk["file"], allow_pickle=False) as data:
                filters = np.array(data["filters"], copy=True)
                record_ids = np.array(data["record_ids"], copy=True)
            if filters.shape != (chunk["count"], self.cfg["filter_length"]) or record_ids.shape != (chunk["count"],):
                raise ValueError(f"Invalid target chunk dimensions: {chunk['file']}.")
            if not np.isfinite(filters).all():
                raise ValueError(f"Nonfinite target coefficients: {chunk['file']}.")
            self.cache[chunk_index] = (filters, record_ids)
            if self.cache_limit is not None and len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(chunk_index)
        filters, record_ids = self.cache[chunk_index]
        row = global_index - chunk["start"]
        if str(record_ids[row]) != record["record_id"]:
            raise ValueError("Target row does not match its sample/environment record.")
        return torch.tensor(filters[row], dtype=torch.float32)

    def __getitem__(self, index):
        global_index, record = self.records[index]
        reference = load_noise(record, self.cfg)
        environment_index = self.environment_indices[record["environment_id"]]
        primary = self.bundle["primary_paths"][environment_index].double().numpy()
        disturbance = causal_numpy(reference, primary)
        observation = reference[:_observation_samples(self.cfg, len(reference))]
        return {
            "observation": torch.tensor(observation, dtype=torch.float32),
            "reference": torch.tensor(reference, dtype=torch.float64),
            "disturbance": torch.tensor(disturbance, dtype=torch.float64),
            "prior_filter": self.bundle["prior_filters"][environment_index].float(),
            "target_filter": self._target(global_index, record),
            "secondary_path": self.bundle["secondary_paths"][environment_index].double(),
            "environment_id": record["environment_id"], "sample_id": record["sample_id"],
            "record_id": record["record_id"], "context": 0,
        }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="Inspect only this many clips per split; mark manifest partial.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = inspect_sfanc(cfg, args.limit)
    print(f"Source counts: {manifest['source_counts']}; paired records={len(manifest['records'])}; partial={manifest['partial']}")
    print(f"Manifest: {cfg['manifest_file']}")


if __name__ == "__main__":
    main()
