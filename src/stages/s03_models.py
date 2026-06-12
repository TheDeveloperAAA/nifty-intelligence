"""Stage 3 — models: walk-forward LightGBM + baselines + conformal intervals.

Per horizon h in {1, 5, 21}: a pooled return regressor and a direction
classifier, evaluated strictly out-of-sample on expanding walk-forward folds
(F1..F6) and a locked test window (2020-01 -> 2021-04). Baselines: zero-return
persistence, expanding per-stock drift, ridge on identical features.

Protocol notes
--------------
- Hyperparameters: one micro-sweep on folds F1-F3 (h=5 regressor), then frozen
  for all horizons/folds/test. Recorded in metrics for transparency.
- Train targets winsorized at train-only percentiles; evaluation unclipped.
- Early stopping on a purged 12-month tail of train; the fold model is then
  REFIT on the full purged train with the chosen iteration count.
- Forecast calibration: for fold k the raw predictions are passed through a
  Mincer-Zarnowitz linear map y = a + b*yhat (b floored at 0) and an isotonic
  probability map, both fitted on the POOLED OUT-OF-SAMPLE predictions of all
  earlier folds — the same prior-fold pooling the conformal layer uses. This is
  leakage-free (earlier folds strictly precede fold k's validation window) and
  far lower-variance than calibrating on a single recent year (which we tried
  and rejected: a one-year intercept imports that year's drift). F1 has no
  prior OOS data and keeps identity calibration. Raw predictions are stored
  alongside calibrated ones for the ablation table.
- Conformal calibration for fold k pools nonconformity scores from earlier
  folds' OOS predictions (F1 seeds from its inner-validation predictions).
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core import conformal as cf
from src.core.indicators import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.core.labeling import winsorize_train
from src.core.model_metrics import direction, price_space, return_space
from src.core.vol_models import ewma_vol_panel
from src.core.walkforward import Fold, fold_masks, parse_folds, purge_cutoff
from src.io_utils import StageCache, read_parquet, write_json, write_parquet

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _lgb_params(cfg, objective: str, seed: int) -> dict:
    p = dict(cfg["model"]["lgbm"])
    n_estimators = p.pop("n_estimators")
    p.pop("early_stopping_rounds", None)
    p.pop("inner_val_months", None)
    p.update(
        objective=objective,
        seed=seed,
        num_threads=cfg.num_threads,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    return p, n_estimators


def _fit_with_inner_es(
    cfg, X, y, dates, train_mask, h: int, calendar, objective: str, seed: int,
    params_override: dict | None = None,
):
    """Early-stop on a purged 12-month tail of train, then refit on full train.

    Returns (booster, best_iter, inner_val_pred_index, inner_val_pred).
    """
    params, n_estimators = _lgb_params(cfg, objective, seed)
    if params_override:
        params.update(params_override)
    es_rounds = cfg["model"]["lgbm"]["early_stopping_rounds"]
    inner_months = cfg["model"]["lgbm"]["inner_val_months"]
    embargo = cfg["cv"]["embargo_days"]

    tr_dates = dates[train_mask]
    train_end = tr_dates.max()
    inner_val_start = train_end - pd.DateOffset(months=inner_months)
    inner_cut = purge_cutoff(calendar, inner_val_start, h, embargo)

    inner_tr = train_mask & (dates <= inner_cut).to_numpy()
    inner_va = train_mask & (dates >= inner_val_start).to_numpy()

    label_ok = np.isfinite(y)
    inner_tr &= label_ok
    inner_va &= label_ok

    dtr = lgb.Dataset(X[inner_tr], label=y[inner_tr],
                      categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)
    dva = lgb.Dataset(X[inner_va], label=y[inner_va], reference=dtr,
                      categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)
    es_booster = lgb.train(
        params, dtr, num_boost_round=n_estimators, valid_sets=[dva],
        callbacks=[lgb.early_stopping(es_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = es_booster.best_iteration or n_estimators

    inner_pred = es_booster.predict(X[inner_va], num_iteration=best_iter)

    full = train_mask & label_ok
    dfull = lgb.Dataset(X[full], label=y[full],
                        categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)
    booster = lgb.train(params, dfull, num_boost_round=best_iter)
    return booster, best_iter, np.where(inner_va)[0], inner_pred


def _ridge_predict(Xn_tr, y_tr, Xn_va, alpha: float):
    pipe = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
    )
    pipe.fit(Xn_tr, y_tr)
    return pipe.predict(Xn_va)


def _mz_calibration(yhat_inner: np.ndarray, y_inner: np.ndarray) -> tuple[float, float]:
    """Mincer-Zarnowitz map fitted on the inner tail: returns (a, b), b >= 0."""
    mask = np.isfinite(yhat_inner) & np.isfinite(y_inner)
    if mask.sum() < 50 or np.std(yhat_inner[mask]) < 1e-12:
        return float(np.nanmean(y_inner)), 0.0
    b, a = np.polyfit(yhat_inner[mask], y_inner[mask], 1)
    if not np.isfinite(b) or b < 0:
        return float(np.nanmean(y_inner[mask])), 0.0
    return float(a), float(b)


def _isotonic_calibration(p_inner: np.ndarray, d_inner: np.ndarray):
    """Isotonic probability map fitted on the inner tail; identity fallback."""
    from sklearn.isotonic import IsotonicRegression

    mask = np.isfinite(p_inner) & np.isfinite(d_inner)
    if mask.sum() < 100 or len(np.unique(d_inner[mask])) < 2:
        return lambda p: p
    iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    iso.fit(p_inner[mask], d_inner[mask])
    return lambda p: iso.predict(p)


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    heavy = cfg.path_for("heavy")
    models_dir = cfg.path_for("models")
    inputs = [heavy / "features_full.parquet", art / "prices_adjusted.parquet"]
    outputs = [
        art / "predictions.parquet",
        art / "model_metrics.json",
        art / "conformal.json",
    ]
    cache = StageCache(art, "models")
    chash = cfg.section_hash("model", "cv", "labels", "conformal", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()
    rng = np.random.default_rng(cfg.stage_seed("models"))

    df = read_parquet(inputs[0])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")

    # EWMA sigma + expanding drift, both causal, from the full adjusted panel
    px = read_parquet(inputs[1])[["date", "symbol", "ret"]]
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    px["sigma"] = ewma_vol_panel(
        px, cfg["volatility"]["ewma_lambda"], cfg["volatility"]["ewma_init_days"]
    )
    px["drift_daily"] = px.groupby("symbol", sort=False)["ret"].transform(
        lambda s: s.expanding(63).mean()
    )
    df = df.merge(px[["date", "symbol", "sigma", "drift_daily"]], on=["date", "symbol"], how="left")

    X = df[FEATURES]
    Xn = df[NUMERIC_FEATURES].to_numpy(np.float32)
    dates = df["date"]
    calendar = np.sort(df["date"].unique())
    feature_start = pd.Timestamp(cfg["cv"]["feature_start"])
    embargo = cfg["cv"]["embargo_days"]
    folds, test_fold = parse_folds(cfg["cv"])
    all_folds: list[Fold] = folds + [test_fold]
    horizons = list(cfg["labels"]["horizons"])
    winsor = tuple(cfg["labels"]["winsor_pct"])
    alpha = cfg["conformal"]["alpha"]
    seed = cfg.seed

    metrics: dict = {"sweep": None, "folds": {}, "pooled_cv": {}, "test": {},
                     "stability": {}, "shuffled_target": {}, "learning_curve": [],
                     "complexity_curve": [], "best_iters": {}}

    # ------------------------------------------------------------------ sweep
    swept_params: dict = {}
    if len(folds) >= 3:
        grid = [
            {"num_leaves": nl, "min_child_samples": mc, "learning_rate": lr}
            for nl in (15, 31) for mc in (100, 200) for lr in (0.03, 0.05)
        ]
        h_sweep = 5
        y_sw = df[f"y_{h_sweep}"].to_numpy(np.float64)
        results = []
        for gparams in grid:
            maes = []
            for fold in folds[:3]:
                tr, va = fold_masks(dates, fold, h_sweep, embargo, calendar, feature_start)
                yw = y_sw.copy()
                yw[tr] = winsorize_train(pd.Series(y_sw[tr]), winsor).to_numpy()
                booster, best_iter, _, _ = _fit_with_inner_es(
                    cfg, X, yw, dates, tr, h_sweep, calendar, "l2", seed, gparams
                )
                pred = booster.predict(X[va], num_iteration=best_iter)
                maes.append(return_space(y_sw[va], pred)["mae"])
            results.append({**gparams, "mean_val_mae": float(np.mean(maes))})
        results.sort(key=lambda r: r["mean_val_mae"])
        best = results[0]
        swept_params = {k: best[k] for k in ("num_leaves", "min_child_samples", "learning_rate")}
        metrics["sweep"] = {"grid_results": results, "chosen": swept_params,
                            "note": "swept on F1-F3 (h=5 regressor) then frozen"}

    # -------------------------------------------------------- main fold loop
    pred_rows: list[pd.DataFrame] = []
    inner_scores: dict[int, np.ndarray] = {}  # h -> F1 inner-val scores
    conformal_meta: dict = {h: {} for h in horizons}

    for h in horizons:
        y = df[f"y_{h}"].to_numpy(np.float64)
        d = df[f"dir_{h}"].to_numpy(np.float64)
        drift_h = df["drift_daily"].to_numpy(np.float64) * h

        for fi, fold in enumerate(all_folds):
            tr, va = fold_masks(dates, fold, h, embargo, calendar, feature_start)
            va &= np.isfinite(y)  # evaluable rows only
            yw = y.copy()
            yw[tr] = winsorize_train(pd.Series(y[tr]), winsor).to_numpy()

            reg, reg_iter, inner_idx, inner_pred = _fit_with_inner_es(
                cfg, X, yw, dates, tr, h, calendar, "l2", seed, swept_params
            )
            clf, clf_iter, _, _ = _fit_with_inner_es(
                cfg, X, d, dates, tr, h, calendar, "binary", seed, swept_params
            )
            metrics["best_iters"][f"h{h}_{fold.name}"] = {"reg": reg_iter, "clf": clf_iter}

            yhat_raw = reg.predict(X[va], num_iteration=reg_iter)
            p_up_raw = clf.predict(X[va], num_iteration=clf_iter)
            ridge_hat = _ridge_predict(
                Xn[tr & np.isfinite(y)], yw[tr & np.isfinite(y)], Xn[va],
                cfg["model"]["ridge_alpha"],
            )

            if fi == 0:
                s_inner = cf.nonconformity(
                    y[inner_idx], inner_pred, df["sigma"].to_numpy()[inner_idx], h
                )
                inner_scores[h] = s_inner[np.isfinite(s_inner)]

            pred_rows.append(pd.DataFrame({
                "date": dates[va].to_numpy(),
                "symbol": df["symbol"][va].to_numpy(),
                "close": df["close"][va].to_numpy(np.float64),
                "horizon": h,
                "fold": fold.name,
                "y_true": y[va],
                "y_pred_raw": yhat_raw,
                "p_up_raw": p_up_raw,
                "dir_true": d[va],
                "yhat_drift": drift_h[va],
                "yhat_ridge": ridge_hat,
                "sigma": df["sigma"][va].to_numpy(np.float64),
            }))

            if fold.name == "TEST":
                models_dir.mkdir(parents=True, exist_ok=True)
                reg.save_model(str(models_dir / f"lgbm_reg_h{h}.txt"))
                clf.save_model(str(models_dir / f"lgbm_dir_h{h}.txt"))

    preds = pd.concat(pred_rows, ignore_index=True)

    # ------------------------------------- calibration on prior-fold OOS data
    preds["y_pred"] = preds["y_pred_raw"]
    preds["p_up"] = preds["p_up_raw"]
    calibration_meta: dict = {}
    fold_order_names = [f.name for f in all_folds]
    for h in horizons:
        calibration_meta[str(h)] = {}
        for fi, fname in enumerate(fold_order_names):
            prior = preds[(preds["horizon"] == h) & (preds["fold"].isin(fold_order_names[:fi]))]
            m = (preds["horizon"] == h) & (preds["fold"] == fname)
            if len(prior) < 500:
                calibration_meta[str(h)][fname] = {"type": "identity", "n_calib": int(len(prior))}
                continue
            a_mz, b_mz = _mz_calibration(
                prior["y_pred_raw"].to_numpy(), prior["y_true"].to_numpy()
            )
            iso_map = _isotonic_calibration(
                prior["p_up_raw"].to_numpy(), prior["dir_true"].to_numpy()
            )
            preds.loc[m, "y_pred"] = a_mz + b_mz * preds.loc[m, "y_pred_raw"]
            preds.loc[m, "p_up"] = np.asarray(
                iso_map(preds.loc[m, "p_up_raw"].to_numpy()), dtype=float
            )
            calibration_meta[str(h)][fname] = {
                "type": "mz+isotonic", "a": a_mz, "b": b_mz, "n_calib": int(len(prior)),
            }
    metrics["calibration"] = calibration_meta

    # ------------------------------------------------------------- conformal
    preds["conf_lo"] = np.nan
    preds["conf_hi"] = np.nan
    fold_order = [f.name for f in all_folds]
    for h in horizons:
        ph = preds[preds["horizon"] == h]
        for fi, fname in enumerate(fold_order):
            prior = ph[ph["fold"].isin(fold_order[:fi])]
            if len(prior) == 0:
                scores = inner_scores.get(h, np.array([]))
                source = "inner_validation"
            else:
                scores = cf.nonconformity(
                    prior["y_true"].to_numpy(), prior["y_pred"].to_numpy(),
                    prior["sigma"].to_numpy(), h,
                )
                source = f"prior_folds<{fname}"
            q = cf.conformal_quantile(scores, alpha)
            m = (preds["horizon"] == h) & (preds["fold"] == fname)
            lo, hi = cf.interval(
                preds.loc[m, "y_pred"].to_numpy(), preds.loc[m, "sigma"].to_numpy(), h, q
            )
            preds.loc[m, "conf_lo"] = lo
            preds.loc[m, "conf_hi"] = hi
            cov = cf.coverage(preds.loc[m, "y_true"].to_numpy(), lo, hi)
            conformal_meta[h][fname] = {
                "q": q, "n_calib": int(len(scores)), "source": source,
                "empirical_coverage": cov,
            }

    # unscaled-conformal ablation + COVID slice coverage on TEST
    for h in horizons:
        ph = preds[(preds["horizon"] == h) & (preds["fold"] != "TEST")]
        q_un = cf.conformal_quantile(
            np.abs(ph["y_true"] - ph["y_pred"]).to_numpy(), alpha
        )
        t = preds[(preds["horizon"] == h) & (preds["fold"] == "TEST")]
        lo_u, hi_u = t["y_pred"] - q_un, t["y_pred"] + q_un
        covid = t[(t["date"] >= "2020-02-01") & (t["date"] <= "2020-04-30")]
        lo_c, hi_c = covid["y_pred"] - q_un, covid["y_pred"] + q_un
        conformal_meta[h]["ablation_unscaled"] = {
            "q": q_un,
            "test_coverage": cf.coverage(t["y_true"].to_numpy(), lo_u.to_numpy(), hi_u.to_numpy()),
            "covid_coverage": cf.coverage(covid["y_true"].to_numpy(), lo_c.to_numpy(), hi_c.to_numpy()),
        }
        covid_scaled = covid
        conformal_meta[h]["covid_slice_scaled"] = {
            "coverage": cf.coverage(
                covid_scaled["y_true"].to_numpy(),
                covid_scaled["conf_lo"].to_numpy(),
                covid_scaled["conf_hi"].to_numpy(),
            ),
            "n": int(len(covid_scaled)),
        }

    # ---------------------------------------------------------------- metrics
    def eval_block(rows: pd.DataFrame, h: int) -> dict:
        y_, yhat = rows["y_true"].to_numpy(), rows["y_pred"].to_numpy()
        naive = np.zeros_like(y_)
        out = {
            "return_space": {
                "model": return_space(y_, yhat),
                "model_raw": return_space(y_, rows["y_pred_raw"].to_numpy()),
                "naive": return_space(y_, naive),
                "drift": return_space(y_, rows["yhat_drift"].to_numpy()),
                "ridge": return_space(y_, rows["yhat_ridge"].to_numpy()),
            },
            "price_space": {
                "model": price_space(rows["close"].to_numpy(), y_, yhat),
                "naive": price_space(rows["close"].to_numpy(), y_, naive),
            },
            "direction": {
                "model": direction(rows["dir_true"].to_numpy(), rows["p_up"].to_numpy(), h),
                "model_raw": direction(rows["dir_true"].to_numpy(), rows["p_up_raw"].to_numpy(), h),
            },
            "conformal_coverage": cf.coverage(
                y_, rows["conf_lo"].to_numpy(), rows["conf_hi"].to_numpy()
            ),
        }
        m = out["return_space"]["model"]["mae"]
        n = out["return_space"]["naive"]["mae"]
        dr = out["return_space"]["drift"]["mae"]
        out["return_space"]["rel_mae_vs_naive"] = float(m / n) if n else float("nan")
        out["return_space"]["rel_mae_vs_drift"] = float(m / dr) if dr else float("nan")
        return out

    for h in horizons:
        metrics["folds"][str(h)] = {}
        for fname in fold_order:
            rows = preds[(preds["horizon"] == h) & (preds["fold"] == fname)]
            block = eval_block(rows, h)
            block["conformal"] = conformal_meta[h].get(fname, {})
            if fname == "TEST":
                metrics["test"][str(h)] = block
            else:
                metrics["folds"][str(h)][fname] = block
        cv_rows = preds[(preds["horizon"] == h) & (preds["fold"] != "TEST")]
        metrics["pooled_cv"][str(h)] = eval_block(cv_rows, h)
        metrics["pooled_cv"][str(h)]["conformal_meta"] = conformal_meta[h]

    # ------------------------------------------------- diagnostics (F3, h=5)
    if len(folds) >= 3:
        h_diag, fold_diag = 5, folds[2]
        y = df[f"y_{h_diag}"].to_numpy(np.float64)
        tr, va = fold_masks(dates, fold_diag, h_diag, embargo, calendar, feature_start)
        yw = y.copy()
        yw[tr] = winsorize_train(pd.Series(y[tr]), winsor).to_numpy()

        # learning curve
        tr_idx = np.where(tr & np.isfinite(y))[0]
        for frac in (0.25, 0.5, 0.75, 1.0):
            sub = rng.choice(tr_idx, size=int(len(tr_idx) * frac), replace=False)
            sub_mask = np.zeros(len(df), dtype=bool)
            sub_mask[sub] = True
            booster, it, _, _ = _fit_with_inner_es(
                cfg, X, yw, dates, sub_mask, h_diag, calendar, "l2", seed, swept_params
            )
            tr_mae = return_space(y[sub_mask], booster.predict(X[sub_mask], num_iteration=it))["mae"]
            va_mae = return_space(y[va], booster.predict(X[va], num_iteration=it))["mae"]
            metrics["learning_curve"].append(
                {"train_frac": frac, "n_train": int(sub_mask.sum()),
                 "train_mae": tr_mae, "val_mae": va_mae}
            )

        # complexity curve
        for nl in (7, 15, 31, 63, 127):
            params_o = dict(swept_params)
            params_o["num_leaves"] = nl
            booster, it, _, _ = _fit_with_inner_es(
                cfg, X, yw, dates, tr, h_diag, calendar, "l2", seed, params_o
            )
            tr_mae = return_space(y[tr], booster.predict(X[tr], num_iteration=it))["mae"]
            va_mae = return_space(y[va], booster.predict(X[va], num_iteration=it))["mae"]
            metrics["complexity_curve"].append(
                {"num_leaves": nl, "train_mae": tr_mae, "val_mae": va_mae}
            )

        # shuffled-target sanity check (affirmative no-leakage evidence)
        y_shuf = yw.copy()
        perm = np.where(tr & np.isfinite(y))[0]
        y_shuf[perm] = rng.permutation(y_shuf[perm])
        booster, it, _, _ = _fit_with_inner_es(
            cfg, X, y_shuf, dates, tr, h_diag, calendar, "l2", seed, swept_params
        )
        shuf_pred = booster.predict(X[va], num_iteration=it)
        rs = return_space(y[va], shuf_pred)
        rs_naive = return_space(y[va], np.zeros(int(va.sum())))
        metrics["shuffled_target"] = {
            "h": h_diag, "fold": fold_diag.name, "val_mae": rs["mae"],
            "val_r2": rs["r2"],
            "rel_mae_vs_naive": float(rs["mae"] / rs_naive["mae"]),
        }

    # ------------------------------------------------ TEST seed stability
    seeds = list(cfg["model"]["stability_seeds"])
    if len(seeds) > 1:
        for h in horizons:
            y = df[f"y_{h}"].to_numpy(np.float64)
            tr, va = fold_masks(dates, test_fold, h, embargo, calendar, feature_start)
            va &= np.isfinite(y)
            yw = y.copy()
            yw[tr] = winsorize_train(pd.Series(y[tr]), winsor).to_numpy()
            maes = []
            for s in seeds:
                booster, it, _, _ = _fit_with_inner_es(
                    cfg, X, yw, dates, tr, h, calendar, "l2", s, swept_params
                )
                maes.append(return_space(y[va], booster.predict(X[va], num_iteration=it))["mae"])
            metrics["stability"][str(h)] = {
                "seeds": seeds, "test_mae_mean": float(np.mean(maes)),
                "test_mae_std": float(np.std(maes)), "test_maes": maes,
            }

    preds["date"] = pd.to_datetime(preds["date"])
    for c in preds.columns:
        if preds[c].dtype == np.float64:
            preds[c] = preds[c].astype(np.float32)
    write_parquet(preds, outputs[0])
    write_json(metrics, outputs[1])
    write_json(conformal_meta, outputs[2])
    cache.record(chash, inputs, outputs, time.time() - t0,
                 extra={"n_predictions": int(len(preds))})
    return False
