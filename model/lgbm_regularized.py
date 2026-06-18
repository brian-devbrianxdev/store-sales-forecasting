"""Regularized per-family LightGBM leg (historically "v8").

What v8 changes vs v7:
  Regularization (closes val→LB gap):
    * num_leaves 128 → 64
    * min_data_in_leaf 200 → 400
    * feature_fraction 0.85 → 0.7
    * lambda_l2 1.0 → 3.0
    * + 5% extra boosting rounds from the probe (was 5% already; kept)

  New features:
    * days_since_first_sale per (store, family) — handles families with
      late launches and stores onboarded mid-history.
    * store_promo_total_{lag1,lag7,lag16} — total onpromotion items in
      the whole store; cannibalisation signal.
    * family_promo_share — share of in-family promotions in the store.
    * promo_x_dow — onpromotion × dow product (catches the "promo lift
      depends on weekday" pattern).
    * sf_promo_uplift_90 — per-(store, family) avg sales on promo vs
      non-promo days over a trailing 90-day window (shifted by h_base).

  CLI:
    * --objective {regression,tweedie}: Tweedie variance_power=1.3 for
      ensemble diversity; metric-aligned for zero-inflated retail series.
    * --seeds: number of seeds per horizon (default 3).
    * --suffix: filename suffix for the submission.
    * --horizons: csv of horizons to train, e.g. "1,8,16" for smoke tests.

  Importable engine:
    * run_engine(cutoffs=...) lets wf_validate.py invoke v8 with arbitrary
      validation cutoffs without re-implementing feature engineering.

Everything else stays from v7 (per-horizon direct, leakage-aware feature
selector, all calendar/holiday/oil/transactions, zero-pair shortcut).
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

HORIZON = 16
TRAIN_FROM_DEFAULT = "2015-01-01"
TEST_START_DEFAULT = pd.Timestamp("2017-08-16")
TEST_END_DEFAULT = pd.Timestamp("2017-08-31")
VAL_START_DEFAULT = pd.Timestamp("2017-07-31")
EARTHQUAKE_DATE = pd.Timestamp("2016-04-16")

LAGS_DAILY = list(range(1, 22))
LAGS_LONG = [28, 35, 49, 63, 91, 119, 182, 364]
ALL_LAGS = LAGS_DAILY + LAGS_LONG
ROLL_WINDOWS = [7, 14, 28, 56, 84, 168]
EWM_HALFLIVES = [7, 28]
HOL_LEADLAG = [-3, -2, -1, 1, 2, 3]
DOW_K_VALUES = [4, 8]
PER_H_ROLL_BASES = [1, 4, 8, 16]

LGB_REG_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=64,
    min_data_in_leaf=400,
    feature_fraction=0.7,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=3.0,
    verbose=-1,
)
LGB_TWEEDIE_PARAMS = dict(LGB_REG_PARAMS,
                          objective="tweedie",
                          tweedie_variance_power=1.3,
                          metric="tweedie")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data() -> dict[str, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA / "stores.csv")
    oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
    hol = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    trans = pd.read_csv(DATA / "transactions.csv", parse_dates=["date"])
    return dict(train=train, test=test, stores=stores, oil=oil,
                holidays=hol, transactions=trans)


# ---------------------------------------------------------------------------
# Holidays (locale-aware) + lead/lag
# ---------------------------------------------------------------------------
def build_holiday_table(hol: pd.DataFrame, stores: pd.DataFrame,
                        end_date: pd.Timestamp) -> pd.DataFrame:
    h = hol.copy()
    h["transferred"] = h["transferred"].astype(bool)
    holiday_eff = h[(h["type"].isin(["Holiday", "Additional", "Bridge", "Transfer"]))
                    & (~h["transferred"])].copy()
    workday = h[h["type"] == "Work Day"].copy()
    event = h[h["type"] == "Event"].copy()

    nat = holiday_eff[holiday_eff["locale"] == "National"][["date"]].drop_duplicates()
    nat["is_holiday_national"] = np.int8(1)
    evt = event[["date"]].drop_duplicates(); evt["is_event_national"] = np.int8(1)
    reg = (holiday_eff[holiday_eff["locale"] == "Regional"][["date", "locale_name"]]
                       .drop_duplicates().rename(columns={"locale_name": "state"}))
    reg["is_holiday_regional"] = np.int8(1)
    loc = (holiday_eff[holiday_eff["locale"] == "Local"][["date", "locale_name"]]
                       .drop_duplicates().rename(columns={"locale_name": "city"}))
    loc["is_holiday_local"] = np.int8(1)
    wd = workday[["date"]].drop_duplicates(); wd["is_workday_override"] = np.int8(1)

    full_dates = pd.date_range("2013-01-01", end_date)
    base = pd.MultiIndex.from_product([full_dates, stores["store_nbr"]],
                                       names=["date", "store_nbr"]).to_frame(index=False)
    base = base.merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left")
    out = base.merge(nat, on="date", how="left")
    out = out.merge(evt, on="date", how="left")
    out = out.merge(wd, on="date", how="left")
    out = out.merge(reg, on=["date", "state"], how="left")
    out = out.merge(loc, on=["date", "city"], how="left")
    for c in ["is_holiday_national", "is_event_national", "is_workday_override",
              "is_holiday_regional", "is_holiday_local"]:
        out[c] = out[c].fillna(0).astype("int8")
    out["is_any_holiday"] = ((out[["is_holiday_national", "is_holiday_regional",
                                    "is_holiday_local"]].sum(axis=1) > 0)
                              & (out["is_workday_override"] == 0)).astype("int8")
    out["is_any_special"] = ((out[["is_holiday_national", "is_holiday_regional",
                                    "is_holiday_local", "is_event_national"]].sum(axis=1) > 0)
                              & (out["is_workday_override"] == 0)).astype("int8")
    return out.drop(columns=["city", "state"])


def add_holiday_leadlag(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    g = panel.groupby("store_nbr", sort=False)["is_any_holiday"]
    for k in HOL_LEADLAG:
        col = f"hol_shift_{'m' if k < 0 else 'p'}{abs(k)}"
        panel[col] = g.shift(-k).fillna(0).astype("int8")
    return panel


# ---------------------------------------------------------------------------
# Oil / Transactions / Calendar
# ---------------------------------------------------------------------------
def build_oil(oil: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    full = pd.DataFrame({"date": pd.date_range(oil["date"].min(), end_date)})
    o = full.merge(oil, on="date", how="left")
    o["dcoilwtico"] = o["dcoilwtico"].ffill().bfill()
    o["oil_ma14"] = o["dcoilwtico"].rolling(14, min_periods=1).mean().astype("float32")
    o["dcoilwtico"] = o["dcoilwtico"].astype("float32")
    return o


def build_transactions(trans: pd.DataFrame, stores: pd.DataFrame,
                       end_date: pd.Timestamp) -> pd.DataFrame:
    full_dates = pd.date_range(trans["date"].min(), end_date)
    base = pd.MultiIndex.from_product([full_dates, stores["store_nbr"]],
                                       names=["date", "store_nbr"]).to_frame(index=False)
    out = base.merge(trans, on=["date", "store_nbr"], how="left")
    out = out.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    out["transactions"] = out["transactions"].astype("float32")
    g = out.groupby("store_nbr", sort=False)["transactions"]
    for lag in [1, 2, 3, 7, 14, 16, 17, 28, 35, 56, 364]:
        out[f"trans_lag_{lag}"] = g.shift(lag).astype("float32")
    for h_b in PER_H_ROLL_BASES:
        out[f"_tb_{h_b}"] = g.shift(h_b).astype("float32")
        gb = out.groupby("store_nbr", sort=False)[f"_tb_{h_b}"]
        for w in [7, 14, 28, 56]:
            out[f"trans_roll_mean_{w}_h{h_b}"] = gb.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            ).astype("float32")
        out.drop(columns=[f"_tb_{h_b}"], inplace=True)
    out["trans_log_lag1"] = np.log1p(out["trans_lag_1"].clip(lower=0)).astype("float32")
    out["trans_log_lag7"] = np.log1p(out["trans_lag_7"].clip(lower=0)).astype("float32")
    out["trans_log_lag16"] = np.log1p(out["trans_lag_16"].clip(lower=0)).astype("float32")
    return out.drop(columns=["transactions"])


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["day"] = d.dt.day.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["year"] = (d.dt.year - 2013).astype("int8")
    df["weekofyear"] = d.dt.isocalendar().week.astype("int8")
    df["quarter"] = d.dt.quarter.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["day_of_month_end"] = d.dt.is_month_end.astype("int8")
    df["day_of_month_start"] = d.dt.is_month_start.astype("int8")
    df["is_wages_day"] = ((df["day"] == 15) | df["day_of_month_end"].astype(bool)).astype("int8")
    df["is_xmas_window"] = (((df["month"] == 12) & (df["day"] >= 23))
                              | ((df["month"] == 1) & (df["day"] <= 2))).astype("int8")
    days_since_eq = (d - EARTHQUAKE_DATE).dt.days
    decay = np.where(days_since_eq.values >= 0,
                     np.exp(-days_since_eq.values / 30.0), 0.0)
    decay = np.where((days_since_eq.values >= 0) & (days_since_eq.values <= 90), decay, 0.0)
    df["earthquake_decay"] = decay.astype("float32")
    return df


# ---------------------------------------------------------------------------
# Lag / rolling features at (store, family) level
# ---------------------------------------------------------------------------
def add_sf_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    g = panel.groupby(["store_nbr", "family"], sort=False)["sales"]
    for lag in ALL_LAGS:
        panel[f"sales_lag_{lag}"] = g.shift(lag).astype("float32")
    return panel


def add_sf_rolling_per_h(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    g = panel.groupby(["store_nbr", "family"], sort=False)["sales"]
    for h_b in PER_H_ROLL_BASES:
        panel[f"_b_{h_b}"] = g.shift(h_b).astype("float32")
        gb = panel.groupby(["store_nbr", "family"], sort=False)[f"_b_{h_b}"]
        for w in ROLL_WINDOWS:
            panel[f"sales_roll_mean_{w}_h{h_b}"] = gb.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            ).astype("float32")
            panel[f"sales_roll_std_{w}_h{h_b}"] = gb.transform(
                lambda s, w=w: s.rolling(w, min_periods=2).std()
            ).astype("float32")
        for w in [14, 28]:
            panel[f"sales_roll_max_{w}_h{h_b}"] = gb.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).max()
            ).astype("float32")
        for hl in EWM_HALFLIVES:
            panel[f"sales_ewm_{hl}_h{h_b}"] = gb.transform(
                lambda s, hl=hl: s.ewm(halflife=hl, min_periods=1).mean()
            ).astype("float32")
        panel.drop(columns=[f"_b_{h_b}"], inplace=True)
    return panel


def add_dow_baseline_per_h(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    for h_b in PER_H_ROLL_BASES:
        panel[f"_b_{h_b}"] = (panel.groupby(["store_nbr", "family"], sort=False)["sales"]
                                      .shift(h_b).astype("float32"))
        g = panel.groupby(["store_nbr", "family", "dow"], sort=False)[f"_b_{h_b}"]
        for k in DOW_K_VALUES:
            panel[f"dow_mean_{k}w_h{h_b}"] = g.transform(
                lambda s, k=k: s.rolling(k, min_periods=1).mean()
            ).astype("float32")
        panel.drop(columns=[f"_b_{h_b}"], inplace=True)
    return panel


def add_aggregate_lag_roll(panel: pd.DataFrame) -> pd.DataFrame:
    fam = (panel.groupby(["family", "date"], observed=True)["sales"]
                  .mean().reset_index(name="fam_mean"))
    fam = fam.sort_values(["family", "date"]).reset_index(drop=True)
    fam["fam_mean_lag16"] = fam.groupby("family")["fam_mean"].shift(HORIZON).astype("float32")
    fam["fam_mean_lag1"] = fam.groupby("family")["fam_mean"].shift(1).astype("float32")
    fam["fam_mean_lag7"] = fam.groupby("family")["fam_mean"].shift(7).astype("float32")
    fam["fam_mean_roll28"] = (
        fam.groupby("family")["fam_mean_lag16"]
            .transform(lambda s: s.rolling(28, min_periods=1).mean())
            .astype("float32"))
    fam = fam.drop(columns=["fam_mean"])
    panel = panel.merge(fam, on=["family", "date"], how="left")

    clu = (panel.groupby(["cluster", "date"], observed=True)["sales"]
                  .mean().reset_index(name="clu_mean"))
    clu = clu.sort_values(["cluster", "date"]).reset_index(drop=True)
    clu["clu_mean_lag16"] = clu.groupby("cluster")["clu_mean"].shift(HORIZON).astype("float32")
    clu["clu_mean_lag1"] = clu.groupby("cluster")["clu_mean"].shift(1).astype("float32")
    clu = clu.drop(columns=["clu_mean"])
    panel = panel.merge(clu, on=["cluster", "date"], how="left")

    eps = 1e-3
    panel["sf_to_fam_ratio"] = (
        (panel["sales_roll_mean_28_h16"] + eps) / (panel["fam_mean_roll28"] + eps)
    ).astype("float32")

    store_total = (panel.groupby(["store_nbr", "date"], observed=True)["sales"]
                          .sum().reset_index(name="store_total"))
    store_total = store_total.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    for k in [1, 7, 16]:
        store_total[f"store_total_lag{k}"] = (
            store_total.groupby("store_nbr")["store_total"].shift(k).astype("float32"))
    store_total = store_total.drop(columns=["store_total"])
    panel = panel.merge(store_total, on=["store_nbr", "date"], how="left")

    for k in [1, 7, 16]:
        panel[f"family_share_lag{k}"] = (
            (panel[f"sales_lag_{k}"] + eps) / (panel[f"store_total_lag{k}"] + eps)
        ).astype("float32")
    return panel


def add_promo_features(panel: pd.DataFrame) -> pd.DataFrame:
    """v7 promo features + v8 store-level promo totals and promo×dow."""
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    g = panel.groupby(["store_nbr", "family"], sort=False)["onpromotion"]
    for k in [1, 7, 14]:
        panel[f"promo_lag_{k}"] = g.shift(k).fillna(0).astype("int32")
    for k in [1, 3, 7, 14]:
        panel[f"promo_lead_{k}"] = g.shift(-k).fillna(0).astype("int32")
    panel["promo_sum_lead16"] = (
        g.transform(lambda s: s.shift(-16).rolling(16, min_periods=1).sum())
          .fillna(0).astype("float32"))
    panel["promo_sum_lag16"] = (
        g.transform(lambda s: s.shift(16).rolling(16, min_periods=1).sum())
          .fillna(0).astype("float32"))
    panel["promo_sum_lag7"] = (
        g.transform(lambda s: s.shift(7).rolling(7, min_periods=1).sum())
          .fillna(0).astype("float32"))

    # NEW: store-level promo total (cannibalisation) — horizon-agnostic
    # because onpromotion is in test.csv for every test row.
    store_promo = (panel.groupby(["store_nbr", "date"], observed=True)["onpromotion"]
                          .sum().reset_index(name="store_promo_total"))
    store_promo = store_promo.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    # current (test-known) total and lags for context
    panel = panel.merge(store_promo, on=["store_nbr", "date"], how="left")
    panel["store_promo_total"] = panel["store_promo_total"].fillna(0).astype("int32")
    for k in [1, 7, 16]:
        panel[f"store_promo_total_lag{k}"] = (
            panel.groupby("store_nbr", sort=False)["store_promo_total"]
                  .shift(k).fillna(0).astype("int32"))

    # NEW: family promo share within store on the *current* day (known at
    # test time since onpromotion is given).
    eps = 1e-3
    panel["family_promo_share"] = (
        (panel["onpromotion"].astype("float32") + eps)
        / (panel["store_promo_total"].astype("float32") + eps)
    ).astype("float32")

    # NEW: promo × dow interaction (current-day, test-known).
    panel["promo_x_dow"] = (panel["onpromotion"].astype("int32")
                              * panel["dow"].astype("int32")).astype("int32")
    return panel


def add_days_since_first_sale(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (store, family), days since first non-zero sale (test-known
    because first-sale date is a property of training history)."""
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    first = (panel[panel["sales"] > 0]
                .groupby(["store_nbr", "family"])["date"].min()
                .reset_index().rename(columns={"date": "first_sale_date"}))
    panel = panel.merge(first, on=["store_nbr", "family"], how="left")
    panel["days_since_first_sale"] = (
        (panel["date"] - panel["first_sale_date"]).dt.days
    ).clip(lower=0).fillna(0).astype("int32")
    panel["is_pre_launch"] = panel["first_sale_date"].isna().astype("int8")
    panel = panel.drop(columns=["first_sale_date"])
    return panel


# ---------------------------------------------------------------------------
# Per-horizon feature-set selection
# ---------------------------------------------------------------------------
SALES_DEPENDENT_LAG_PREFIXES = (
    "sales_lag_", "trans_lag_", "trans_log_lag",
    "fam_mean_lag", "clu_mean_lag",
    "store_total_lag", "family_share_lag",
    # store_promo_total_lag is store-level onpromotion, which IS known in
    # the test window — but historical filling depends on past data only,
    # so still treat as horizon-agnostic. Don't add here.
)
ROLL_PER_H_PREFIXES = ("sales_roll_", "sales_ewm_", "dow_mean_", "trans_roll_mean_")


def _h_base_for(h: int) -> int:
    eligible = [b for b in PER_H_ROLL_BASES if b >= h]
    return min(eligible) if eligible else max(PER_H_ROLL_BASES)


def _trailing_lag_k(name: str) -> int | None:
    s = name
    n = len(s)
    while n > 0 and s[n - 1].isdigit():
        n -= 1
    if n == len(s):
        return None
    tail = s[n:]
    head = s[:n]
    if head.endswith("_lag") or head.endswith("_lag_"):
        try:
            return int(tail)
        except ValueError:
            return None
    return None


def select_features_for_h(all_feature_cols: list[str], h: int) -> list[str]:
    h_base = _h_base_for(h)
    keep: list[str] = []
    for c in all_feature_cols:
        if any(c.startswith(p) for p in SALES_DEPENDENT_LAG_PREFIXES):
            k = _trailing_lag_k(c)
            if k is not None and k >= h:
                keep.append(c)
            continue
        if any(c.startswith(p) for p in ROLL_PER_H_PREFIXES):
            if c.endswith(f"_h{h_base}"):
                keep.append(c)
            continue
        keep.append(c)
    return keep


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _params_for(objective: str, *, quantile_alpha: float | None = None) -> dict:
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
    """Returns per-row sample weights (float32).

    decay_halflife_days: if set, weight *= 0.5 ** ((anchor - date) / halflife).
    august_boost: multiplier for rows where month == 8.
    september_boost: same for month == 9.
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


def rmsle(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    y_true_raw = np.clip(y_true_raw, 0, None)
    y_pred_raw = np.clip(y_pred_raw, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true_raw) - np.log1p(y_pred_raw)) ** 2)))


# ---------------------------------------------------------------------------
# Reusable feature-building pipeline
# ---------------------------------------------------------------------------
@dataclass
class Cutoffs:
    train_from: pd.Timestamp = pd.Timestamp(TRAIN_FROM_DEFAULT)
    val_start: pd.Timestamp = VAL_START_DEFAULT
    test_start: pd.Timestamp = TEST_START_DEFAULT
    test_end: pd.Timestamp = TEST_END_DEFAULT


def build_panel(cuts: Cutoffs,
                 *, store_types: list[str] | None = None,
                 umap_path: Path | str | None = None,
                 ) -> tuple[pd.DataFrame, set[tuple], list[str], list[str]]:
    """Returns (full_panel_with_all_features, zero_set, feature_cols, cat_cols).

    The panel covers data from train.csv (clipped to cuts.train_from on the
    train side, NOT on feature-engineering side — we use the full history
    for lags/rolls) plus a synthetic future window from cuts.test_start to
    cuts.test_end with sales=NaN and onpromotion taken from the test.csv if
    available (for the real test) or fabricated as zeros (for WF holdouts
    that go beyond train.csv).

    store_types: if provided, restrict to stores whose `type` is in this list
    (e.g. ["C"]). Useful for per-cluster specialist models.
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

    # Build the "future" frame for the WF/test window. If real test (i.e.
    # cuts.test_start == 2017-08-16) we use test.csv directly. Otherwise we
    # synthesise: take train rows for [cuts.test_start, cuts.test_end] and
    # keep them with their actual sales (so we can score the WF window).
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

    truth_col = full["_truth"] if "_truth" in full.columns else None

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
    full = full.merge(hol_panel, on=["date", "store_nbr"], how="left")
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
    horizons: list[int] = list(range(1, HORIZON + 1)),
    probe_h: int = HORIZON,
    fixed_iter_mult: float = 1.05,
    skip_refit: bool = False,
    decay_halflife_days: float | None = None,
    august_boost: float = 1.0,
    september_boost: float = 1.0,
    store_types: list[str] | None = None,
    quantile_alpha: float | None = None,
    umap_path: Path | str | None = None,
) -> dict:
    """Train per-horizon models. Returns dict with predictions and diags.

    skip_refit=True: only train on FIT (pre-val) and predict val; do NOT
    refit on FIT+VAL for the test window. Useful for fast WF scoring where
    we only want the val score.
    """
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["regression", "tweedie", "quantile"],
                     default="regression")
    ap.add_argument("--quantile-alpha", type=float, default=None,
                     help="quantile level for objective=quantile (0..1)")
    ap.add_argument("--umap", default=None,
                     help="path to UMAP embeddings parquet (store_nbr,family,umap_*)")
    ap.add_argument("--seeds", type=int, default=3,
                     help="seeds per horizon (default 3)")
    ap.add_argument("--horizons", default=None,
                     help="csv of horizons to train (default: 1..16). Useful for smoke tests.")
    ap.add_argument("--suffix", default=None, help="submission filename suffix")
    ap.add_argument("--train-from", default=TRAIN_FROM_DEFAULT)
    ap.add_argument("--skip-refit", action="store_true",
                     help="skip refit on train+val (val score only)")
    ap.add_argument("--decay-halflife", type=float, default=None,
                     help="time-decay halflife in days (e.g. 365 → 1yr halflife). "
                           "Default off.")
    ap.add_argument("--august-boost", type=float, default=1.0,
                     help="multiplier for August training rows (default 1.0)")
    ap.add_argument("--september-boost", type=float, default=1.0,
                     help="multiplier for September training rows (default 1.0)")
    ap.add_argument("--store-types", default=None,
                     help="comma-separated store types to keep (e.g. 'C'). "
                           "Default: all 5 types A..E.")
    args = ap.parse_args()

    seeds_pool = [42, 1337, 2026, 7, 99]
    seeds = seeds_pool[: args.seeds]
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
    # Internal val-window RMSLE is a sanity check only; the leg's standalone sigma
    # is the Kaggle score of submission_v8_{suffix}.csv (notebook §4.2).
    print(f"Expected RMSLE ~ {result['val_rmsle']:.5f} (val-window sanity check)", flush=True)


if __name__ == "__main__":
    main()
