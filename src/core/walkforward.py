"""Expanding walk-forward folds with horizon purge + embargo.

A label at t covers (t, t+h]. Training rows whose label window could touch the
validation period are dropped: keep train rows with date <= the (h + embargo)-th
trading day before val_start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def parse_folds(cv_cfg: dict) -> tuple[list[Fold], Fold]:
    folds = [
        Fold(f["name"], pd.Timestamp(f["train_end"]), pd.Timestamp(f["val_start"]),
             pd.Timestamp(f["val_end"]))
        for f in cv_cfg["folds"]
    ]
    t = cv_cfg["test"]
    test = Fold(t["name"], pd.Timestamp(t["train_end"]), pd.Timestamp(t["val_start"]),
                pd.Timestamp(t["val_end"]))
    return folds, test


def purge_cutoff(calendar: np.ndarray, val_start: pd.Timestamp, h: int, embargo: int) -> pd.Timestamp:
    """Last allowed train date: (h + embargo) trading days before val_start."""
    calendar = np.asarray(calendar, dtype="datetime64[ns]")
    idx = np.searchsorted(calendar, np.datetime64(val_start))
    cut = max(idx - (h + embargo) - 1, 0)
    return pd.Timestamp(calendar[cut])


def fold_masks(
    dates: pd.Series,
    fold: Fold,
    h: int,
    embargo: int,
    calendar: np.ndarray,
    feature_start: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    cutoff = purge_cutoff(calendar, fold.val_start, h, embargo)
    train = (dates >= feature_start) & (dates <= min(cutoff, fold.train_end))
    val = (dates >= fold.val_start) & (dates <= fold.val_end)
    return train.to_numpy(), val.to_numpy()
