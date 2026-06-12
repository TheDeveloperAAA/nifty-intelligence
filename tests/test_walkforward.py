"""Fold construction: disjoint windows, embargo >= horizon + configured gap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.walkforward import Fold, fold_masks, parse_folds, purge_cutoff

CV_CFG = {
    "feature_start": "2001-07-02",
    "folds": [
        {"name": "F1", "train_end": "2007-12-31", "val_start": "2008-01-01", "val_end": "2009-12-31"},
        {"name": "F2", "train_end": "2009-12-31", "val_start": "2010-01-01", "val_end": "2011-12-31"},
    ],
    "test": {"name": "TEST", "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2021-04-30"},
    "embargo_days": 5,
}


def _calendar():
    return pd.bdate_range("2001-07-02", "2021-04-30").to_numpy()


def test_parse_folds():
    folds, test = parse_folds(CV_CFG)
    assert [f.name for f in folds] == ["F1", "F2"]
    assert test.name == "TEST"
    assert folds[0].val_start > folds[0].train_end


def test_purge_cutoff_gap():
    cal = _calendar()
    h, embargo = 21, 5
    cut = purge_cutoff(cal, pd.Timestamp("2008-01-01"), h, embargo)
    # the number of trading days strictly between cutoff and val_start >= h+embargo
    n_between = ((cal > np.datetime64(cut)) & (cal < np.datetime64("2008-01-01"))).sum()
    assert n_between >= h + embargo


def test_fold_masks_disjoint_and_ordered():
    cal = _calendar()
    dates = pd.Series(np.repeat(cal, 2))  # two symbols per day
    folds, test = parse_folds(CV_CFG)
    h = 21
    masks = [fold_masks(dates, f, h, 5, cal, pd.Timestamp("2001-07-02")) for f in folds + [test]]
    for tr, va in masks:
        assert not (tr & va).any(), "train/val overlap"
        assert dates[tr].max() < dates[va].min(), "train must precede validation"
    # validation windows of different folds are disjoint
    v1 = set(dates[masks[0][1]].unique())
    v2 = set(dates[masks[1][1]].unique())
    assert v1.isdisjoint(v2)


def test_label_window_cannot_touch_validation():
    cal = _calendar()
    dates = pd.Series(cal)
    fold = Fold("F1", pd.Timestamp("2007-12-31"), pd.Timestamp("2008-01-01"),
                pd.Timestamp("2009-12-31"))
    h, embargo = 21, 5
    tr, va = fold_masks(dates, fold, h, embargo, cal, pd.Timestamp("2001-07-02"))
    last_train = dates[tr].max()
    # a label at last_train spans (t, t+21]; even shifted by the embargo it
    # must end before the validation start
    idx = np.searchsorted(cal, np.datetime64(last_train))
    label_end = cal[idx + h]
    assert pd.Timestamp(label_end) < fold.val_start
