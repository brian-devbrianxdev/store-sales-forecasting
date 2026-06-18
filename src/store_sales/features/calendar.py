"""Calendar, holiday, oil and transaction features (shared exogenous signals).

These builders attach time-based and exogenous covariates to a panel keyed by
``(date, store_nbr[, family])``. The module-level constants are bound from
``config.yaml`` at import time, so the function bodies are unchanged from the
original ``lgbm_regularized.py`` while configuration stays centralized.

:func:`national_holiday_dates` is the single national-holiday helper that
replaces the three near-identical implementations the legs previously carried.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_config

_cfg = get_config()
EARTHQUAKE_DATE: pd.Timestamp = _cfg.common.earthquake_date
HOL_LEADLAG: list[int] = list(_cfg.lgbm_v8.hol_leadlag)
PER_H_ROLL_BASES: list[int] = list(_cfg.lgbm_v8.per_h_roll_bases)


def national_holiday_dates(holidays: pd.DataFrame) -> set[pd.Timestamp]:
    """National-level special-day dates (active, non-transferred, non-work-day).

    A single locale-agnostic helper used by the Chronos and neural legs. A date
    qualifies when it has an active (``transferred == False``) National-locale
    Holiday/Additional/Bridge/Event/Transfer row that is not a ``Work Day``.

    Args:
        holidays: The raw ``holidays_events.csv`` frame (``date`` parsed).

    Returns:
        Set of national special-day timestamps.
    """
    active = holidays[(holidays["transferred"] == False)  # noqa: E712
                      & (holidays["type"] != "Work Day")]
    national = active[active["locale"] == "National"]
    return set(national["date"])


def build_holiday_table(hol: pd.DataFrame, stores: pd.DataFrame,
                        end_date: pd.Timestamp) -> pd.DataFrame:
    """Locale-aware per-(date, store) holiday flag table.

    Builds national/regional/local holiday, national-event, and work-day-override
    indicators on a full ``(date × store)`` grid, then derives ``is_any_holiday``
    and ``is_any_special`` composites.

    Args:
        hol: Raw holidays/events frame.
        stores: Stores metadata (provides ``city``/``state`` for locale joins).
        end_date: Last date to materialize the grid through.

    Returns:
        A frame keyed by ``(date, store_nbr)`` with the holiday-flag columns
        (``city``/``state`` dropped).
    """
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
    """Add holiday lead/lag indicator columns per store.

    For each offset ``k`` in :data:`HOL_LEADLAG`, shifts ``is_any_holiday`` within
    each store to flag the days surrounding a holiday.

    Args:
        panel: Panel containing ``store_nbr``, ``date``, ``is_any_holiday``.

    Returns:
        The panel with ``hol_shift_{m,p}{k}`` columns added (store/date sorted).
    """
    panel = panel.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    g = panel.groupby("store_nbr", sort=False)["is_any_holiday"]
    for k in HOL_LEADLAG:
        col = f"hol_shift_{'m' if k < 0 else 'p'}{abs(k)}"
        panel[col] = g.shift(-k).fillna(0).astype("int8")
    return panel


def build_oil(oil: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """Daily oil price (forward/back-filled) plus a 14-day moving average.

    Args:
        oil: Raw ``oil.csv`` frame.
        end_date: Last date to materialize through.

    Returns:
        A ``(date, dcoilwtico, oil_ma14)`` frame on a continuous daily index.
    """
    full = pd.DataFrame({"date": pd.date_range(oil["date"].min(), end_date)})
    o = full.merge(oil, on="date", how="left")
    o["dcoilwtico"] = o["dcoilwtico"].ffill().bfill()
    o["oil_ma14"] = o["dcoilwtico"].rolling(14, min_periods=1).mean().astype("float32")
    o["dcoilwtico"] = o["dcoilwtico"].astype("float32")
    return o


def build_transactions(trans: pd.DataFrame, stores: pd.DataFrame,
                       end_date: pd.Timestamp) -> pd.DataFrame:
    """Per-store transaction lags and per-horizon rolling means.

    Builds a full ``(date × store)`` grid, then derives transaction lags, log
    lags, and rolling means anchored at each per-horizon base in
    :data:`PER_H_ROLL_BASES`.

    Args:
        trans: Raw ``transactions.csv`` frame.
        stores: Stores metadata (defines the store universe).
        end_date: Last date to materialize through.

    Returns:
        A frame keyed by ``(date, store_nbr)`` with the transaction features
        (the raw ``transactions`` column dropped).
    """
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
    """Add calendar features and an earthquake decay term.

    Derives day-of-week/month/year/quarter, weekend/month-edge/wages-day/Xmas
    flags, and an exponential decay (30-day scale, 0–90 day support) from the
    2016 earthquake date in :data:`EARTHQUAKE_DATE`.

    Args:
        df: Panel with a ``date`` column.

    Returns:
        The panel with calendar columns added (mutated in place and returned).
    """
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
