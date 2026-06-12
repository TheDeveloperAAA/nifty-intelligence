"""Stage 8 — explainability: exact TreeSHAP via LightGBM pred_contrib.

No `shap` import in the computation path (numpy-compat safety): LightGBM's
pred_contrib=True returns exact TreeSHAP values. The shap package is used only
to render the beeswarm figure, with a matplotlib fallback.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.explain_text import feature_label, render_reason
from src.core.indicators import ALL_FEATURES, CATEGORICAL_FEATURES
from src.io_utils import StageCache, read_parquet, write_json, write_parquet


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    heavy = cfg.path_for("heavy")
    models_dir = cfg.path_for("models")
    horizons = list(cfg["labels"]["horizons"])
    inputs = [heavy / "features_full.parquet"] + [
        models_dir / f"lgbm_reg_h{h}.txt" for h in horizons
    ]
    outputs = [
        art / "shap_global.csv",
        art / "shap_sample.parquet",
        art / "shap_latest.parquet",
        art / "shap_local_examples.json",
    ]
    cache = StageCache(art, "explain")
    chash = cfg.section_hash("explain", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()
    rng = np.random.default_rng(cfg.stage_seed("explain"))

    df = read_parquet(inputs[0])
    df["date"] = pd.to_datetime(df["date"])
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    X = df[ALL_FEATURES]

    n_sample = int(cfg["explain"]["shap_sample_rows"])
    # stratify the sample by year so no era dominates the beeswarm
    years = df["date"].dt.year
    idx_parts = []
    for _, grp in df.groupby(years):
        take = max(1, int(n_sample * len(grp) / len(df)))
        idx_parts.append(rng.choice(grp.index.to_numpy(), size=min(take, len(grp)),
                                    replace=False))
    sample_idx = np.sort(np.concatenate(idx_parts))[:n_sample]
    Xs = X.loc[sample_idx]

    global_rows, sample_frames = [], []
    latest_rows, local_examples = [], {}
    for h in horizons:
        booster = lgb.Booster(model_file=str(models_dir / f"lgbm_reg_h{h}.txt"))
        contrib = booster.predict(Xs, pred_contrib=True)
        shap_vals = contrib[:, :-1]  # last column is the expected value
        mean_abs = np.abs(shap_vals).mean(axis=0)
        for f, v in zip(ALL_FEATURES, mean_abs):
            global_rows.append({"horizon": h, "feature": f, "label": feature_label(f),
                                "mean_abs_shap": float(v)})

        top_idx = np.argsort(mean_abs)[::-1][:12]
        frame = pd.DataFrame({
            "horizon": h,
            "date": df.loc[sample_idx, "date"].to_numpy(),
            "symbol": df.loc[sample_idx, "symbol"].to_numpy(),
        })
        for j in top_idx:
            f = ALL_FEATURES[j]
            vals = Xs[f]
            if f in CATEGORICAL_FEATURES:
                vals = vals.cat.codes
            frame[f"val_{f}"] = vals.to_numpy(dtype=np.float32)
            frame[f"shap_{f}"] = shap_vals[:, j].astype(np.float32)
        sample_frames.append(frame)

        # local explanations for every symbol at its latest available date
        last_rows = df.loc[df.groupby("symbol")["date"].idxmax()]
        contrib_last = booster.predict(last_rows[ALL_FEATURES], pred_contrib=True)
        base = contrib_last[:, -1]
        sv = contrib_last[:, :-1]
        for r, (i, row) in enumerate(last_rows.iterrows()):
            order = np.argsort(np.abs(sv[r]))[::-1][: cfg["explain"]["top_features_local"]]
            for rank, j in enumerate(order):
                f = ALL_FEATURES[j]
                raw_val = row[f]
                fval = float(raw_val) if f not in CATEGORICAL_FEATURES else float("nan")
                latest_rows.append({
                    "symbol": row["symbol"], "horizon": h, "date": row["date"],
                    "rank": rank, "feature": f, "label": feature_label(f),
                    "value": fval, "shap": float(sv[r, j]),
                    "reason": render_reason(f, fval if fval == fval else 0.0,
                                            float(sv[r, j]), h),
                    "base_value": float(base[r]),
                    "prediction": float(base[r] + sv[r].sum()),
                })

        # one narrated example per horizon: deepest-COVID date for RELIANCE
        ex_sym = "RELIANCE" if (df["symbol"] == "RELIANCE").any() else df["symbol"].iloc[0]
        ex = df[(df["symbol"] == ex_sym) & (df["date"] == "2020-03-23")]
        if len(ex) == 0:
            ex = df[df["symbol"] == ex_sym].tail(1)
        c = booster.predict(ex[ALL_FEATURES], pred_contrib=True)[0]
        order = np.argsort(np.abs(c[:-1]))[::-1][:8]
        local_examples[str(h)] = {
            "symbol": ex_sym,
            "date": str(ex["date"].iloc[0].date()),
            "base_value": float(c[-1]),
            "prediction": float(c.sum()),
            "contributions": [
                {"feature": ALL_FEATURES[j], "label": feature_label(ALL_FEATURES[j]),
                 "value": (float(ex[ALL_FEATURES[j]].iloc[0])
                           if ALL_FEATURES[j] not in CATEGORICAL_FEATURES else None),
                 "shap": float(c[j])}
                for j in order
            ],
        }

    pd.DataFrame(global_rows).to_csv(outputs[0], index=False)
    write_parquet(pd.concat(sample_frames, ignore_index=True), outputs[1])
    write_parquet(pd.DataFrame(latest_rows), outputs[2])
    write_json(local_examples, outputs[3])
    cache.record(chash, inputs, outputs, time.time() - t0)
    return False
