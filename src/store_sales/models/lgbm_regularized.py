"""Regularized per-family LightGBM leg (historically "v8").

What v8 changes vs v7:
  Regularization (closes val→LB gap):
    * num_leaves 128 → 64
    * min_data_in_leaf 200 → 400
    * feature_fraction 0.85 → 0.7
    * lambda_l2 1.0 → 3.0
    * + 5% extra boosting rounds from the probe

  New features:
    * days_since_first_sale per (store, family)
    * store_promo_total_{lag1,lag7,lag16} — store-level promo cannibalisation
    * family_promo_share — share of in-family promotions in the store
    * promo_x_dow — onpromotion × dow interaction
    * sf_to_fam_ratio / family aggregate lags

Feature engineering lives in :mod:`store_sales.features.calendar` and
:mod:`store_sales.features.lgbm_features`. Hyperparameters, lag lists and cutoff
dates come from ``config.yaml``. ``run_engine(cutoffs=...)`` is importable so a
walk-forward harness can drive v8 with arbitrary validation cutoffs, and
:mod:`store_sales.models.catboost_family` reuses :func:`build_panel` and
:func:`select_features_for_h` for identical features.
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from .. import paths
from ..config import get_config
from ..io.data_loading import load_raw_frames
from ..metrics import rmsle
from ..features.calendar import (
    add_calendar,
    add_holiday_leadlag,
    build_holiday_table,
    build_oil,
    build_transactions,
)
from ..features.common_features import (
    add_oil_dynamics,
    holiday_extra_frame,
)
from ..features.lgbm_features import (
    add_aggregate_lag_roll,
    add_days_since_first_sale,
    add_dow_baseline_per_h,
    add_promo_features,
    add_sf_lag_features,
    add_sf_rolling_per_h,
    select_features_for_h,
    _h_base_for,
)

warnings.filterwarnings("ignore")

_cfg = get_config()

DATA = paths.DATA
OUT = paths.SUBMISSIONS

HORIZON = _cfg.common.horizon
TRAIN_FROM_DEFAULT = str(_cfg.common.train_from.date())
TEST_START_DEFAULT = _cfg.common.test_start
TEST_END_DEFAULT = _cfg.common.test_end
VAL_START_DEFAULT = _cfg.common.val_start

SEEDS_POOL = list(_cfg.lgbm_v8.seeds_pool)
FIXED_ITER_MULT_DEFAULT = _cfg.lgbm_v8.fixed_iter_mult

LGB_REG_PARAMS = dict(_cfg.lgbm_v8.params)
LGB_TWEEDIE_PARAMS = dict(LGB_REG_PARAMS, **_cfg.lgbm_v8.tweedie)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _params_for(objective: str, *, quantile_alpha: float | None = None) -> dict:
    """Return the LightGBM params dict for an objective.

    Args:
        objective: One of ``"regression"``, ``"tweedie"``, ``"quantile"``.
        quantile_alpha: Required quantile level when ``objective == "quantile"``.

    Returns:
        A params dict ready for ``lgb.train``.

    Raises:
        ValueError: For an unknown objective or a missing quantile alpha.
    """
    if objective == "tweedie":
        return dict(LGB_TWEEDIE_PARAMS)
    if objective == "regression":
        return dict(LGB_REG_PARAMS)
    if objective == "quantile":
        if quantile_alpha is None:
            raise ValueError("quantile objective requires --quantile-alpha")
        return dict(LGB_REG_PARAMS, objective="quantile", metric="quantile",
                    alpha=quantile_alpha)
    raise ValueError(f"unknown objective: {objective!r}")


def train_lgb_es(X_fit, y_fit, X_val, y_val, cat_cols, *, seed, objective,
                  num_boost_round=2500, log_period=400, patience=120,
                  weight_fit=None, weight_val=None, quantile_alpha=None):
    """Train a LightGBM booster with early stopping on a validation set.

    Returns:
        The trained ``lgb.Booster`` (best iteration retained).
    """
    params = dict(_params_for(objective, quantile_alpha=quantile_alpha),
                  seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    dtrain = lgb.Dataset(X_fit, label=y_fit, weight=weight_fit,
                          categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, weight=weight_val,
                        categorical_feature=cat_cols,
                        reference=dtrain, free_raw_data=False)
    return lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(patience), lgb.log_evaluation(log_period)],
    )


def train_lgb_fixed(X_fit, y_fit, cat_cols, *, seed, objective, num_boost_round,
                     weight=None, quantile_alpha=None):
    """Train a LightGBM booster for a fixed number of rounds (no early stop).

    Returns:
        The trained ``lgb.Booster``.
    """
    params = dict(_params_for(objective, quantile_alpha=quantile_alpha),
                  seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    dtrain = lgb.Dataset(X_fit, label=y_fit, weight=weight,
                          categorical_feature=cat_cols, free_raw_data=False)
    return lgb.train(params, dtrain, num_boost_round=num_boost_round)


def compute_sample_weight(
    dates: pd.Series, *,
    decay_halflife_days: float | None = None,
    august_boost: float = 1.0,
    september_boost: float = 1.0,
    anchor_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """Per-row sample weights from optional time decay and month boosts.

    Args:
        dates: Row dates.
        decay_halflife_days: If set, ``weight *= 0.5 ** ((anchor - date)/halflife)``.
        august_boost: Multiplier for rows where ``month == 8``.
        september_boost: Multiplier for rows where ``month == 9``.
        anchor_date: Reference date for the decay (defaults to ``dates.max()``).

    Returns:
        ``float32`` weights, shape ``(len(dates),)``.
    """
    n = len(dates)
    w = np.ones(n, dtype="float32")
    if august_boost != 1.0:
        mask = (dates.dt.month == 8).values
        w[mask] *= float(august_boost)
    if september_boost != 1.0:
        mask = (dates.dt.month == 9).values
        w[mask] *= float(september_boost)
    if decay_halflife_days is not None and decay_halflife_days > 0:
        if anchor_date is None:
            anchor_date = dates.max()
        days_back = (anchor_date - dates).dt.days.values
        w *= np.power(0.5, days_back / float(decay_halflife_days)).astype("float32")
    return w


# ---------------------------------------------------------------------------
# Reusable feature-building pipeline
# ---------------------------------------------------------------------------
@dataclass
class Cutoffs:
    """Train/val/test date cutoffs for one fit. Defaults come from config."""

    train_from: pd.Timestamp = pd.Timestamp(TRAIN_FROM_DEFAULT)
    val_start: pd.Timestamp = VAL_START_DEFAULT
    test_start: pd.Timestamp = TEST_START_DEFAULT
    test_end: pd.Timestamp = TEST_END_DEFAULT


def load_data() -> dict[str, pd.DataFrame]:
    """Load the raw competition frames (delegates to :func:`load_raw_frames`)."""
    return load_raw_frames()


def build_panel(cuts: Cutoffs,
                 *, store_types: list[str] | None = None,
                 umap_path: Path | str | None = None,
                 ) -> tuple[pd.DataFrame, set[tuple], list[str], list[str]]:
    """Build the full feature panel for the v8 leg.

    Returns ``(full_panel_with_all_features, zero_set, feature_cols, cat_cols)``.

    The panel covers train.csv history (full history for lags/rolls) plus a
    synthetic future window ``[test_start, test_end]`` (sales=NaN; onpromotion
    from test.csv for the real test, or from train for walk-forward holdouts).

    Args:
        cuts: Date cutoffs.
        store_types: If given, restrict to stores whose ``type`` is in this list.
        umap_path: Optional parquet of per-(store, family) UMAP embeddings.

    Returns:
        ``(full, zero_set, feature_cols, cat_cols)``.
    """
    d = load_data()
    train = d["train"]
    test = d["test"]
    stores = d["stores"]

    if store_types:
        keep_stores = stores[stores["type"].isin(store_types)]["store_nbr"].tolist()
        print(f"Filtering to store types {store_types}: {len(keep_stores)}/{len(stores)} stores",
               flush=True)
        stores = stores[stores["store_nbr"].isin(keep_stores)].reset_index(drop=True)
        train = train[train["store_nbr"].isin(keep_stores)].reset_index(drop=True)
        test = test[test["store_nbr"].isin(keep_stores)].reset_index(drop=True)
    oil = build_oil(d["oil"], cuts.test_end)
    hol_panel = build_holiday_table(d["holidays"], stores, cuts.test_end)
    trans_panel = build_transactions(d["transactions"], stores, cuts.test_end)

    # Detect zero pairs from data available STRICTLY BEFORE test_start, so
    # WF runs don't peek into the future.
    train_pre = train[train["date"] < cuts.test_start]
    zero_keys = (train_pre.groupby(["store_nbr", "family"])["sales"]
                           .apply(lambda s: (s == 0).all()))
    zero_pairs = zero_keys[zero_keys].index
    zero_set = set(map(tuple, zero_pairs.to_list()))
    print(f"Always-zero pairs (pre-{cuts.test_start.date()}): {len(zero_set)}", flush=True)

    train["sales"] = train["sales"].astype("float32")

    # Build the "future" frame for the WF/test window.
    if cuts.test_start >= train["date"].max() + pd.Timedelta(days=1):
        # real test window — sales unknown
        future = test[["date", "store_nbr", "family", "onpromotion"]].copy()
        future["sales"] = np.nan
        # train piece: everything before test_start
        past = train[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    else:
        # WF: pretend [test_start..test_end] is future, but keep ground
        # truth around in a separate column for scoring later.
        future_mask = (train["date"] >= cuts.test_start) & (train["date"] <= cuts.test_end)
        past = train.loc[~future_mask, ["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
        future = train.loc[future_mask, ["date", "store_nbr", "family", "onpromotion"]].copy()
        future["sales"] = np.nan
        # we'll also keep ground truth aside:
        future["_truth"] = train.loc[future_mask, "sales"].clip(lower=0).values

    full = pd.concat([past, future], ignore_index=True)
    full["onpromotion"] = full["onpromotion"].fillna(0).astype("int32")

    if zero_set:
        zp_df = pd.DataFrame(list(zero_set), columns=["store_nbr", "family"])
        zp_df["_drop"] = 1
        full = full.merge(zp_df, on=["store_nbr", "family"], how="left")
        kept = full["_drop"].isna()
        print(f"Rows before zero-pair drop: {len(full):,}; after: {int(kept.sum()):,}",
               flush=True)
        full = full[kept].drop(columns=["_drop"]).reset_index(drop=True)

    full = full.merge(stores, on="store_nbr", how="left")
    if umap_path is not None:
        umap_df = pd.read_parquet(umap_path)
        umap_cols = [c for c in umap_df.columns if c.startswith("umap_")]
        full = full.merge(umap_df[["store_nbr", "family"] + umap_cols],
                            on=["store_nbr", "family"], how="left")
        for c in umap_cols:
            full[c] = full[c].fillna(0).astype("float32")
        print(f"  merged UMAP embeddings: {umap_cols} ({len(umap_df)} pairs)", flush=True)
    full = full.merge(oil, on="date", how="left")
    # Add advanced oil dynamics
    oil_dyn = add_oil_dynamics(oil, price_col="dcoilwtico")
    dyn_cols = [c for c in oil_dyn.columns if c.startswith("oil_")]
    full = full.merge(oil_dyn[["date"] + dyn_cols], on="date", how="left")
    
    full = full.merge(hol_panel, on=["date", "store_nbr"], how="left")
    # Add advanced holiday distance features
    cal = pd.date_range(cuts.train_from, cuts.test_end, freq="D")
    hol_extra = holiday_extra_frame(cal, d["holidays"])
    full = full.merge(hol_extra, on="date", how="left")
    
    full = full.merge(trans_panel, on=["date", "store_nbr"], how="left")
    for c in ["is_holiday_national", "is_event_national", "is_workday_override",
              "is_holiday_regional", "is_holiday_local",
              "is_any_holiday", "is_any_special"]:
        full[c] = full[c].fillna(0).astype("int8")

    full = add_calendar(full)
    full = add_holiday_leadlag(full)

    print("  add_sf_lag_features...", flush=True)
    full = add_sf_lag_features(full)
    print("  add_sf_rolling_per_h...", flush=True)
    full = add_sf_rolling_per_h(full)
    print("  add_dow_baseline_per_h...", flush=True)
    full = add_dow_baseline_per_h(full)
    print("  add_aggregate_lag_roll...", flush=True)
    full = add_aggregate_lag_roll(full)
    print("  add_promo_features...", flush=True)
    full = add_promo_features(full)
    print("  add_days_since_first_sale...", flush=True)
    full = add_days_since_first_sale(full)

    cat_cols = ["store_nbr", "family", "city", "state", "type", "cluster"]
    for c in cat_cols:
        full[c] = full[c].astype("category")

    full["target_log"] = np.log1p(full["sales"].clip(lower=0))

    drop_cols = {"date", "sales", "target_log", "_truth"}
    feature_cols = [c for c in full.columns if c not in drop_cols]
    print(f"Total feature pool: {len(feature_cols)}", flush=True)
    return full, zero_set, feature_cols, cat_cols


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def run_engine(
    cuts: Cutoffs,
    *,
    objective: str,
    seeds: list[int],
    horizons: list[int] | None = None,
    probe_h: int = HORIZON,
    fixed_iter_mult: float = FIXED_ITER_MULT_DEFAULT,
    skip_refit: bool = False,
    decay_halflife_days: float | None = None,
    august_boost: float = 1.0,
    september_boost: float = 1.0,
    store_types: list[str] | None = None,
    quantile_alpha: float | None = None,
    umap_path: Path | str | None = None,
) -> dict:
    """Train per-horizon models and return predictions + diagnostics.

    Args:
        cuts: Date cutoffs.
        objective: ``regression`` | ``tweedie`` | ``quantile``.
        seeds: Seeds to average per horizon.
        horizons: Horizons to train (default 1..HORIZON).
        probe_h: Horizon used to probe the boosting-round count.
        fixed_iter_mult: Multiplier on the probed best iteration.
        skip_refit: If True, only fit on FIT (pre-val) and predict val.
        decay_halflife_days, august_boost, september_boost: Sample weighting.
        store_types: Optional store-type restriction.
        quantile_alpha: Quantile level for the quantile objective.
        umap_path: Optional UMAP-embedding parquet.

    Returns:
        Dict with val/test predictions, keys, RMSLE diagnostics, and the panel.
    """
    if horizons is None:
        horizons = list(range(1, HORIZON + 1))
    t0 = time.time()
    full, zero_set, feature_cols, cat_cols_all = build_panel(cuts, store_types=store_types,
                                                                 umap_path=umap_path)

    is_test = full["date"] >= cuts.test_start
    is_test_in_window = is_test & (full["date"] <= cuts.test_end)
    train_mask = (~is_test) & (full["date"] >= cuts.train_from) & full["sales"].notna()
    val_mask = train_mask & (full["date"] >= cuts.val_start) & (full["date"] < cuts.test_start)
    fit_mask = train_mask & (full["date"] < cuts.val_start)

    val_offset = ((full.loc[val_mask, "date"] - cuts.val_start).dt.days + 1).astype(int).values
    test_offset = ((full.loc[is_test_in_window, "date"] - cuts.test_start).dt.days + 1).astype(int).values

    # Tweedie needs raw target; regression uses log1p target.
    is_tweedie = (objective == "tweedie")
    y_fit_log = full.loc[fit_mask, "target_log"].values
    y_val_log = full.loc[val_mask, "target_log"].values
    y_val_raw = full.loc[val_mask, "sales"].clip(lower=0).values
    y_fit_raw = full.loc[fit_mask, "sales"].clip(lower=0).values
    y_train_log = full.loc[train_mask, "target_log"].values
    y_train_raw = full.loc[train_mask, "sales"].clip(lower=0).values

    y_fit_used = y_fit_raw if is_tweedie else y_fit_log
    y_val_used = y_val_raw if is_tweedie else y_val_log
    y_train_used = y_train_raw if is_tweedie else y_train_log

    # Sample weights (used for both probe+per-h fit and final refit).
    use_weights = (decay_halflife_days is not None or
                    august_boost != 1.0 or september_boost != 1.0)
    if use_weights:
        anchor = cuts.test_start
        w_fit = compute_sample_weight(
            full.loc[fit_mask, "date"],
            decay_halflife_days=decay_halflife_days,
            august_boost=august_boost,
            september_boost=september_boost,
            anchor_date=anchor,
        )
        w_train = compute_sample_weight(
            full.loc[train_mask, "date"],
            decay_halflife_days=decay_halflife_days,
            august_boost=august_boost,
            september_boost=september_boost,
            anchor_date=anchor,
        )
        print(f"Sample weights: halflife={decay_halflife_days}, "
              f"aug_boost={august_boost}, sep_boost={september_boost}", flush=True)
        print(f"  w_fit  min={w_fit.min():.3f} max={w_fit.max():.3f} mean={w_fit.mean():.3f}",
               flush=True)
    else:
        w_fit = None
        w_train = None

    n_val = int(val_mask.sum())
    n_test = int(is_test_in_window.sum())
    print(f"Train rows: {int(fit_mask.sum()):,}  Val rows: {n_val:,}  Test rows: {n_test:,}",
           flush=True)

    # Probe for iter count
    print(f"\n=== Probe (h={probe_h}, ES, seed={seeds[0]}, obj={objective}) ===", flush=True)
    feat_probe = select_features_for_h(feature_cols, probe_h)
    cat_probe = [c for c in cat_cols_all if c in feat_probe]
    Xp_fit = full.loc[fit_mask, feat_probe]
    Xp_val = full.loc[val_mask, feat_probe]
    probe_model = train_lgb_es(Xp_fit, y_fit_used, Xp_val, y_val_used, cat_probe,
                                  seed=seeds[0], objective=objective,
                                  num_boost_round=3000, patience=150,
                                  weight_fit=w_fit, weight_val=None,
                                  quantile_alpha=quantile_alpha)
    best_iter = int(probe_model.best_iteration or 1500)
    fixed_iter = max(int(best_iter * fixed_iter_mult), 300)
    print(f"Probe best_iter={best_iter}, fixed_iter={fixed_iter}", flush=True)

    val_pred_log = np.zeros(n_val, dtype=np.float64)
    test_pred_log = np.zeros(n_test, dtype=np.float64) if n_test else None

    print(f"\n=== Per-horizon ({len(horizons)} horizons, {len(seeds)} seeds) ===", flush=True)
    X_test_all = full.loc[is_test_in_window] if n_test else None

    per_h_val_rmsle: dict[int, float] = {}
    for h in horizons:
        feat_h = select_features_for_h(feature_cols, h)
        cat_h = [c for c in cat_cols_all if c in feat_h]
        print(f"\n-- h={h}: {len(feat_h)} features, h_base={_h_base_for(h)}", flush=True)

        X_fit_h = full.loc[fit_mask, feat_h]
        X_val_h_full = full.loc[val_mask, feat_h]
        val_rows_h = (val_offset == h)
        n_val_h = int(val_rows_h.sum())
        if n_val_h == 0:
            print(f"   no val rows for h={h} — skipping", flush=True)
            continue

        seed_val = np.zeros(n_val_h, dtype=np.float64)
        for s in seeds:
            m = train_lgb_fixed(X_fit_h, y_fit_used, cat_h, seed=s,
                                  objective=objective, num_boost_round=fixed_iter,
                                  weight=w_fit, quantile_alpha=quantile_alpha)
            preds = m.predict(X_val_h_full.iloc[val_rows_h])
            if is_tweedie:
                preds = np.log1p(np.clip(preds, 0, None))  # to log space
            seed_val += preds / len(seeds)
        idx_val = np.where(val_rows_h)[0]
        val_pred_log[idx_val] = seed_val

        y_val_raw_h = y_val_raw[val_rows_h]
        pred_raw_h = np.clip(np.expm1(seed_val), 0, None)
        h_rmsle = rmsle(y_val_raw_h, pred_raw_h)
        per_h_val_rmsle[h] = h_rmsle
        print(f"   h={h} val RMSLE: {h_rmsle:.5f}  elapsed={time.time() - t0:.0f}s",
               flush=True)

        if skip_refit or n_test == 0:
            continue
        X_full_h = full.loc[train_mask, feat_h]
        test_rows_h = (test_offset == h)
        if int(test_rows_h.sum()) == 0:
            continue
        X_test_h = X_test_all.iloc[test_rows_h][feat_h]
        seed_test = np.zeros(int(test_rows_h.sum()), dtype=np.float64)
        for s in seeds:
            m = train_lgb_fixed(X_full_h, y_train_used, cat_h, seed=s,
                                  objective=objective, num_boost_round=fixed_iter,
                                  weight=w_train, quantile_alpha=quantile_alpha)
            preds = m.predict(X_test_h)
            if is_tweedie:
                preds = np.log1p(np.clip(preds, 0, None))
            seed_test += preds / len(seeds)
        idx_test = np.where(test_rows_h)[0]
        test_pred_log[idx_test] = seed_test

    val_pred_raw = np.clip(np.expm1(val_pred_log), 0, None)
    final_val_rmsle = rmsle(y_val_raw, val_pred_raw)
    print(f"\n=== Overall val RMSLE: {final_val_rmsle:.5f} ===", flush=True)

    val_keys = full.loc[val_mask, ["date", "store_nbr", "family"]].reset_index(drop=True)
    val_keys["forecast_offset"] = val_offset

    return dict(
        val_pred_log=val_pred_log,
        val_pred_raw=val_pred_raw,
        val_truth_raw=y_val_raw,
        val_keys=val_keys,
        val_rmsle=final_val_rmsle,
        per_h_val_rmsle=per_h_val_rmsle,
        test_pred_log=test_pred_log,
        full=full,
        is_test_in_window=is_test_in_window,
        zero_set=zero_set,
        fixed_iter=fixed_iter,
    )


def write_submission(result: dict, *, suffix: str) -> Path:
    """Write the v8 test predictions to ``submission_v8_{suffix}.csv``.

    Args:
        result: The dict returned by :func:`run_engine`.
        suffix: Filename suffix.

    Returns:
        Path to the written submission.
    """
    full = result["full"]
    is_test = result["is_test_in_window"]
    test_pred_raw = np.clip(np.expm1(result["test_pred_log"]), 0, None)

    test_csv = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    sub_kept = full.loc[is_test, ["date", "store_nbr", "family"]].copy()
    sub_kept["sales"] = test_pred_raw
    sub = test_csv[["id", "date", "store_nbr", "family"]].merge(
        sub_kept, on=["date", "store_nbr", "family"], how="left")
    sub["sales"] = sub["sales"].fillna(0.0)
    assert sub["sales"].notna().all(), "missing predictions"
    submission = sub[["id", "sales"]].sort_values("id")
    out_path = OUT / f"submission_v8_{suffix}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved {out_path} (rows={len(submission)})", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the v8 LightGBM leg."""
    ap = argparse.ArgumentParser(description="Regularized per-family LightGBM (v8) leg")
    ap.add_argument("--objective", choices=["regression", "tweedie", "quantile"],
                     default="regression")
    ap.add_argument("--quantile-alpha", type=float, default=None,
                     help="quantile level for objective=quantile (0..1)")
    ap.add_argument("--umap", default=None,
                     help="path to UMAP embeddings parquet (store_nbr,family,umap_*)")
    ap.add_argument("--seeds", type=int, default=_cfg.lgbm_v8.default_seeds,
                     help="seeds per horizon")
    ap.add_argument("--horizons", default=None,
                     help="csv of horizons to train (default: 1..16). Useful for smoke tests.")
    ap.add_argument("--suffix", default=None, help="submission filename suffix")
    ap.add_argument("--train-from", default=TRAIN_FROM_DEFAULT)
    ap.add_argument("--skip-refit", action="store_true",
                     help="skip refit on train+val (val score only)")
    ap.add_argument("--decay-halflife", type=float, default=None,
                     help="time-decay halflife in days (e.g. 365 → 1yr halflife). Default off.")
    ap.add_argument("--august-boost", type=float, default=1.0,
                     help="multiplier for August training rows (default 1.0)")
    ap.add_argument("--september-boost", type=float, default=1.0,
                     help="multiplier for September training rows (default 1.0)")
    ap.add_argument("--store-types", default=None,
                     help="comma-separated store types to keep (e.g. 'C'). Default: all.")
    args = ap.parse_args(argv)

    seeds = SEEDS_POOL[: args.seeds]
    horizons = ([int(x) for x in args.horizons.split(",")]
                 if args.horizons else list(range(1, HORIZON + 1)))
    suffix = args.suffix or args.objective

    cuts = Cutoffs(
        train_from=pd.Timestamp(args.train_from),
        val_start=VAL_START_DEFAULT,
        test_start=TEST_START_DEFAULT,
        test_end=TEST_END_DEFAULT,
    )
    print(f"Cutoffs: train_from={cuts.train_from.date()} val={cuts.val_start.date()} "
          f"test=[{cuts.test_start.date()}..{cuts.test_end.date()}]", flush=True)
    print(f"Objective={args.objective} seeds={seeds} horizons={horizons}", flush=True)

    store_types = ([s.strip() for s in args.store_types.split(",")]
                    if args.store_types else None)
    result = run_engine(cuts, objective=args.objective, seeds=seeds,
                          horizons=horizons, skip_refit=args.skip_refit,
                          decay_halflife_days=args.decay_halflife,
                          august_boost=args.august_boost,
                          september_boost=args.september_boost,
                          store_types=store_types,
                          quantile_alpha=args.quantile_alpha,
                          umap_path=args.umap)
    if args.skip_refit or result["test_pred_log"] is None:
        return
    write_submission(result, suffix=suffix)
    print(f"Expected RMSLE ~ {result['val_rmsle']:.5f} (val-window sanity check)", flush=True)


if __name__ == "__main__":
    main()
