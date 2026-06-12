"""Configuration loading and derived settings.

config.yaml is the single source of truth; this module only parses it,
resolves paths relative to the repo root and derives per-stage seeds.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    root: Path = field(default=REPO_ROOT)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def seed(self) -> int:
        return int(self.raw["run"]["seed"])

    @property
    def num_threads(self) -> int:
        return int(self.raw["run"]["num_threads"])

    def stage_seed(self, stage: str) -> int:
        """Deterministic per-stage seed derived from the global seed."""
        digest = hashlib.sha256(f"{self.seed}:{stage}".encode()).hexdigest()
        return int(digest[:8], 16)

    def path_for(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    def section_hash(self, *sections: str) -> str:
        """Hash of selected config sections; used for stage-cache invalidation."""
        payload = yaml.safe_dump(
            {s: self.raw.get(s) for s in sections}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=cfg_path)
