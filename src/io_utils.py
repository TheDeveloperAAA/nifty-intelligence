"""Artifact I/O and stage caching.

Every pipeline stage writes its outputs plus a manifest
(artifacts/_manifests/<stage>.json) recording config-section and input hashes.
A stage is skipped when its manifest matches the current state and all outputs
exist, unless --force is given.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
    return path


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(f"not JSON serializable: {type(o)}")


class StageCache:
    """Hash-manifest cache deciding whether a stage can be skipped."""

    def __init__(self, artifacts_dir: Path, stage: str):
        self.stage = stage
        self.manifest_path = artifacts_dir / "_manifests" / f"{stage}.json"

    def is_fresh(self, config_hash: str, inputs: list[Path], outputs: list[Path]) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            manifest = read_json(self.manifest_path)
        except (json.JSONDecodeError, OSError):
            return False
        if manifest.get("config_hash") != config_hash:
            return False
        recorded = manifest.get("input_hashes", {})
        current = {str(p): sha256_file(p) for p in inputs if p.exists()}
        if recorded != current:
            return False
        return all(p.exists() for p in outputs)

    def record(
        self,
        config_hash: str,
        inputs: list[Path],
        outputs: list[Path],
        wall_time: float,
        extra: dict | None = None,
    ) -> None:
        manifest = {
            "stage": self.stage,
            "config_hash": config_hash,
            "input_hashes": {str(p): sha256_file(p) for p in inputs if p.exists()},
            "output_hashes": {str(p): sha256_file(p) for p in outputs if p.exists()},
            "wall_time_s": round(wall_time, 2),
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            manifest.update(extra)
        write_json(manifest, self.manifest_path)
