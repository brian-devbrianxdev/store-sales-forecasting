"""CatBoost per-horizon family member for ensemble diversity.

Reuses `lgbm_regularized`'s `build_panel` and `select_features_for_h` for
identical features and leakage-aware per-horizon feature filtering. The model
is CatBoostRegressor with RMSE loss on log1p(sales) — aligned with RMSLE.

Why CatBoost: genuinely different inductive bias from LightGBM (oblivious
trees, different categorical handling, different boosting dynamics). Adds
decorrelated errors that compound with the LightGBM legs in blends.

Usage:
    python src/catboost_family.py --seeds 2 --suffix cat
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import lgbm_regularized as lgbm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"

HORIZON = lgbm.HORIZON


def cat_params(seed: int) -> dict:
    # Restored to depth=6 + iterations=2500 per plan B.1 (2026-05-18).
    # Smoke at depth=6 had h=16 val 0.416 vs v8 0.430; full run pursued
    # for orthogonal-bias diversification.
    return dict(
        iterations=2500,
        depth=6,
        l2_leaf_reg=5.0,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        thread_count=-1,
        verbose=300,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        rsm=0.7,
    )


def train_cat_es(X_fit, y_fit, X_val, y_val, cat_cols, *, seed,
                  iterations=1500, patience=120):
    p = cat_params(seed) | dict(iterations=iterations,
                                  early_stopping_rounds=patience)
    cat_idx = [X_fit.columns.get_loc(c) for c in cat_cols if c in X_fit.columns]
    train_pool = Pool(X_fit, y_fit, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    m = CatBoostRegressor(**p)
    m.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return m


def train_cat_fixed(X_fit, y_fit, cat_cols, *, seed, iterations):
    p = cat_params(seed) | dict(iterations=iterations)
    cat_idx = [X_fit.columns.get_loc(c) for c in cat_cols if c in X_fit.columns]
    pool = Pool(X_fit, y_fit, cat_features=cat_idx)
    m = CatBoostRegressor(**p)
    m.fit(pool)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2,
                     help="seeds per horizon (default 2 — catboost slower than lgbm)")
    ap.add_argument("--horizons", default=None,
                     help="csv subset, e.g. '1,8,16' for smoke test")
    ap.add_argument("--suffix", default="cat")
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args()

    seeds_pool = [42, 1337, 2026, 7, 99]
    seeds = seeds_pool[: args.seeds]
    horizons = ([int(x) for x in args.horizons.split(",")]
                 if args.horizons else list(range(1, HORIZON + 1)))

    t0 = time.time()
    cuts = lgbm.Cutoffs(
        train_from=pd.Timestamp(lgbm.TRAIN_FROM_DEFAULT),
        val_start=lgbm.VAL_START_DEFAULT,
        test_start=lgbm.TEST_START_DEFAULT,
        test_end=lgbm.TEST_END_DEFAULT,
    )
    full, zero_set, feature_cols, cat_cols_all = lgbm.build_panel(cuts)

    # IMPORTANT: CatBoost requires categorical columns to be int or string,
    # not pandas category. Convert "category" cols to integer codes, but
    # also preserve the original string values for downstream merges/OOF.
    full["_family_str"] = full["family"].astype(str)
    full["_store_str"] = full["store_nbr"].astype(int).astype(str)
    for c in cat_cols_all:
        if full[c].dtype.name == "category":
            full[c] = full[c].cat.codes.astype("int32")

    is_test = full["date"] >= cuts.test_start
    is_test_in_window = is_test & (full["date"] <= cuts.test_end)
    train_mask = (~is_test) & (full["date"] >= cuts.train_from) & full["sales"].notna()
    val_mask = train_mask & (full["date"] >= cuts.val_start) & (full["date"] < cuts.test_start)
    fit_mask = train_mask & (full["date"] < cuts.val_start)

    val_offset = ((full.loc[val_mask, "date"] - cuts.val_start).dt.days + 1).astype(int).values
    test_offset = ((full.loc[is_test_in_window, "date"] - cuts.test_start).dt.days + 1).astype(int).values

    y_fit_log = full.loc[fit_mask, "target_log"].values
    y_val_log = full.loc[val_mask, "target_log"].values
    y_val_raw = full.loc[val_mask, "sales"].clip(lower=0).values
    y_train_log = full.loc[train_mask, "target_log"].values

    n_val = int(val_mask.sum())
    n_test = int(is_test_in_window.sum())
    print(f"Train rows: {int(fit_mask.sum()):,}  Val rows: {n_val:,}  Test rows: {n_test:,}",
           flush=True)

    # Probe on h=16 for iter count
    print(f"\n=== Probe (h={HORIZON}, ES, seed={seeds[0]}) ===", flush=True)
    feat_probe = lgbm.select_features_for_h(feature_cols, HORIZON)
    cat_probe = [c for c in cat_cols_all if c in feat_probe]
    Xp_fit = full.loc[fit_mask, feat_probe]
    Xp_val = full.loc[val_mask, feat_probe]
    probe = train_cat_es(Xp_fit, y_fit_log, Xp_val, y_val_log, cat_probe,
                          seed=seeds[0], iterations=3000, patience=150)
    best_iter = probe.best_iteration_ or 2500
    fixed_iter = max(int(best_iter * 1.05), 500)
    print(f"Probe best_iter={best_iter}, fixed_iter={fixed_iter}", flush=True)

    val_pred_log = np.zeros(n_val, dtype=np.float64)
    test_pred_log = np.zeros(n_test, dtype=np.float64) if n_test else None
    per_h_val_rmsle: dict[int, float] = {}

    X_test_all = full.loc[is_test_in_window] if n_test else None

    for h in horizons:
        feat_h = lgbm.select_features_for_h(feature_cols, h)
        cat_h = [c for c in cat_cols_all if c in feat_h]
        print(f"\n-- h={h}: {len(feat_h)} features, h_base={lgbm._h_base_for(h)}", flush=True)
        X_fit_h = full.loc[fit_mask, feat_h]
        X_val_h_full = full.loc[val_mask, feat_h]
        val_rows_h = (val_offset == h)
        n_val_h = int(val_rows_h.sum())
        if n_val_h == 0:
            continue

        seed_val = np.zeros(n_val_h, dtype=np.float64)
        for s in seeds:
            m = train_cat_fixed(X_fit_h, y_fit_log, cat_h, seed=s, iterations=fixed_iter)
            seed_val += m.predict(X_val_h_full.iloc[val_rows_h]) / len(seeds)
        idx_val = np.where(val_rows_h)[0]
        val_pred_log[idx_val] = seed_val
        pred_raw_h = np.clip(np.expm1(seed_val), 0, None)
        h_rmsle = lgbm.rmsle(y_val_raw[val_rows_h], pred_raw_h)
        per_h_val_rmsle[h] = h_rmsle
        print(f"   h={h} val RMSLE: {h_rmsle:.5f}  elapsed={time.time() - t0:.0f}s", flush=True)

        if args.skip_refit or n_test == 0:
            continue
        X_full_h = full.loc[train_mask, feat_h]
        test_rows_h = (test_offset == h)
        if int(test_rows_h.sum()) == 0:
            continue
        X_test_h = X_test_all.iloc[test_rows_h][feat_h]
        seed_test = np.zeros(int(test_rows_h.sum()), dtype=np.float64)
        for s in seeds:
            m = train_cat_fixed(X_full_h, y_train_log, cat_h, seed=s, iterations=fixed_iter)
            seed_test += m.predict(X_test_h) / len(seeds)
        idx_test = np.where(test_rows_h)[0]
        test_pred_log[idx_test] = seed_test

    val_pred_raw = np.clip(np.expm1(val_pred_log), 0, None)
    final = lgbm.rmsle(y_val_raw, val_pred_raw)
    print(f"\n=== Overall val RMSLE: {final:.5f} ===", flush=True)

    if args.skip_refit or n_test == 0:
        return

    # Write submission. Use the preserved string family/store, since the
    # categorical version was converted to int codes for CatBoost.
    test_pred_raw = np.clip(np.expm1(test_pred_log), 0, None)
    test_csv = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    sub_kept = full.loc[is_test_in_window, ["date", "_store_str", "_family_str"]].copy()
    sub_kept = sub_kept.rename(columns={"_store_str": "store_nbr",
                                          "_family_str": "family"})
    sub_kept["store_nbr"] = sub_kept["store_nbr"].astype(int)
    sub_kept["sales"] = test_pred_raw
    sub = test_csv[["id", "date", "store_nbr", "family"]].merge(
        sub_kept, on=["date", "store_nbr", "family"], how="left")
    sub["sales"] = sub["sales"].fillna(0.0)
    submission = sub[["id", "sales"]].sort_values("id")
    out_path = OUT / f"submission_{args.suffix}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
