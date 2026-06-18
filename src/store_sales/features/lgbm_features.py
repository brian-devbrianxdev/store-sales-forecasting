"""Per-(store, family) lag/rolling features and the per-horizon feature selector.

These builders create the autoregressive feature pool consumed by the LightGBM
v8 and CatBoost legs. :func:`select_features_for_h` enforces leakage-aware
feature selection: for a forecast offset ``h`` only lags ``>= h`` and the
matching per-horizon rolling base are kept.

Module-level constants are bound from ``config.yaml`` at import time so the
function bodies are byte-identical to the original ``lgbm_regularized.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_config

_cfg = get_config()
HORIZON: int = _cfg.common.horizon
ALL_LAGS: list[int] = _cfg.lgbm_v8.all_lags
ROLL_WINDOWS: list[int] = list(_cfg.lgbm_v8.roll_windows)
EWM_HALFLIVES: list[int] = list(_cfg.lgbm_v8.ewm_halflives)
DOW_K_VALUES: list[int] = list(_cfg.lgbm_v8.dow_k_values)
PER_H_ROLL_BASES: list[int] = list(_cfg.lgbm_v8.per_h_roll_bases)

# Lag-feature prefixes whose trailing integer is a true lag in days; these are
# filtered by `select_features_for_h` so horizon h only sees lags >= h.
SALES_DEPENDENT_LAG_PREFIXES = (
    "sales_lag_", "trans_lag_", "trans_log_lag",
    "fam_mean_lag", "clu_mean_lag",
    "store_total_lag", "family_share_lag",
    # store_promo_total_lag is store-level onpromotion, which IS known in
    # the test window — but historical filling depends on past data only,
    # so still treat as horizon-agnostic. Don't add here.
)
ROLL_PER_H_PREFIXES = ("sales_roll_", "sales_ewm_", "dow_mean_", "trans_roll_mean_")


def add_sf_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-(store, family) sales lag columns for every lag in :data:`ALL_LAGS`.

    Args:
        panel: Panel with ``store_nbr``, ``family``, ``date``, ``sales``.

    Returns:
        The panel (store/family/date sorted) with ``sales_lag_{k}`` columns.
    """
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    g = panel.groupby(["store_nbr", "family"], sort=False)["sales"]
    for lag in ALL_LAGS:
        panel[f"sales_lag_{lag}"] = g.shift(lag).astype("float32")
    return panel


def add_sf_rolling_per_h(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-horizon rolling mean/std/max and EWM features of sales.

    For each base ``h_b`` in :data:`PER_H_ROLL_BASES`, sales are shifted by
    ``h_b`` (so horizon ``h`` only uses information available ``h_b`` days back),
    then rolled over :data:`ROLL_WINDOWS` and EWM-smoothed over
    :data:`EWM_HALFLIVES`.

    Args:
        panel: Panel with ``store_nbr``, ``family``, ``date``, ``sales``.

    Returns:
        The panel with ``sales_roll_*_h{h_b}`` and ``sales_ewm_*_h{h_b}`` columns.
    """
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
    """Add per-horizon day-of-week rolling baselines.

    For each base ``h_b`` and each window ``k`` in :data:`DOW_K_VALUES`, computes
    the rolling mean of shifted sales within each ``(store, family, dow)`` group.

    Args:
        panel: Panel with ``store_nbr``, ``family``, ``date``, ``sales``, ``dow``.

    Returns:
        The panel with ``dow_mean_{k}w_h{h_b}`` columns.
    """
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
    """Add family / cluster / store aggregate lag and rolling features.

    Builds family-mean and cluster-mean lagged series, a store-total lag set,
    the per-family share of store total, and the family-relative sales ratio.

    Args:
        panel: Panel with ``family``, ``cluster``, ``store_nbr``, ``date``,
            ``sales`` and the ``sales_roll_mean_28_h16`` / ``sales_lag_{k}``
            columns produced by the rolling/lag builders.

    Returns:
        The panel with the aggregate features merged in.
    """
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
    """v7 promo features + v8 store-level promo totals and promo×dow.

    Adds promo lag/lead columns, leading/trailing promo sums, the store-level
    promo total and its lags (cannibalisation), the per-family promo share, and
    the promo×day-of-week interaction.

    Args:
        panel: Panel with ``store_nbr``, ``family``, ``date``, ``onpromotion``,
            ``dow``.

    Returns:
        The panel with the promo features added.
    """
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
    """Days since each (store, family)'s first non-zero sale.

    Test-known because the first-sale date is a property of training history.
    Also flags pairs that never sold (``is_pre_launch``).

    Args:
        panel: Panel with ``store_nbr``, ``family``, ``date``, ``sales``.

    Returns:
        The panel with ``days_since_first_sale`` and ``is_pre_launch`` columns.
    """
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


def _h_base_for(h: int) -> int:
    """Return the smallest per-horizon rolling base ``>= h`` (else the largest)."""
    eligible = [b for b in PER_H_ROLL_BASES if b >= h]
    return min(eligible) if eligible else max(PER_H_ROLL_BASES)


def _trailing_lag_k(name: str) -> int | None:
    """Parse the trailing lag integer ``k`` from a ``*_lag[_]{k}`` column name.

    Args:
        name: Feature column name.

    Returns:
        The lag ``k`` if ``name`` ends in ``_lag{k}``/``_lag_{k}``, else ``None``.
    """
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
    """Leakage-aware feature subset for forecast offset ``h``.

    Keeps lag features only when their lag ``>= h``, keeps per-horizon rolling
    features only when they match the base ``_h{h_base}`` for this ``h``, and
    keeps all horizon-agnostic features.

    Args:
        all_feature_cols: The full feature pool.
        h: Forecast offset (1-based; 1..horizon).

    Returns:
        The subset of feature columns valid for horizon ``h``.
    """
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
