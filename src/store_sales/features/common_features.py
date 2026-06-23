"""Leakage-safe exogenous features shared across legs (oil + holidays).

These builders were first written inline in :mod:`store_sales.models.darts_family`
and are lifted here so the neural leg (TSMixer/TiDE) can reuse *exactly* the same
audited logic. Every feature is **date-level** (independent of store/family), so
the public API returns ``date``-keyed frames that each leg merges with
``merge(on="date")`` — fully decoupled from how a leg represents its panel.

Leakage policy (single source of truth):
  * Oil dynamics are deterministic, strictly backward-looking transforms of a
    *fully observed* daily WTI series (oil.csv runs through the test window), so
    they are valid future covariates. Warm-up NaNs sit only at the leading edge.
  * Holiday features are properties of the published calendar (known well past
    the test window), so "days to the NEXT special day" is legitimate
    future-covariate information — never derived from ``sales``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Standardised output column names (re-exported by the legs so their feature
# lists stay in sync with this module).
OIL_DYNAMIC_COLS = [
    "oil_ret_7", "oil_ret_28",
    "oil_lag_16", "oil_lag_28", "oil_lag_56",
    "oil_vol_28",
]
HOLIDAY_EXTRA_COLS = [
    "is_transferred_origin", "days_to_next_special", "days_since_prev_special",
]
# Clip holiday distances to a shopping-relevant ±window (keeps them bounded for
# the downstream Scaler and avoids huge sentinels at the calendar edges).
SPECIAL_DIST_CAP = 30


# -------------------- Oil --------------------

def add_oil_dynamics(oil_daily: pd.DataFrame, price_col: str = "oil") -> pd.DataFrame:
    """Append :data:`OIL_DYNAMIC_COLS` to a daily, fully-filled oil price frame.

    Args:
        oil_daily: Frame sorted by ``date`` on a contiguous daily index whose
            ``price_col`` has **no NaN** (caller is responsible for the gap
            filling, which differs per leg — ``interpolate(time)`` vs ``ffill``).
        price_col: Name of the price column (``"oil"`` for darts, ``"dcoilwtico"``
            for the neural leg).

    Returns:
        ``oil_daily`` with 7/28-day returns, 16/28/56-day lags, and a 28-day
        realised-volatility regime (rolling std of daily returns) added.

    Leakage note: ``shift``/``pct_change``/``rolling`` only look backward. The
    warm-up NaNs they produce live solely at the 2013 leading edge — lags carry
    the earliest known price backwards (``bfill``), returns/vol get 0. Because
    the price has no interior/trailing gaps, that ``bfill`` can never pull a
    test-window value into earlier dates.
    """
    price = oil_daily[price_col]
    daily_ret = price.pct_change(fill_method=None)
    oil_daily["oil_ret_7"] = price.pct_change(7, fill_method=None)
    oil_daily["oil_ret_28"] = price.pct_change(28, fill_method=None)
    oil_daily["oil_lag_16"] = price.shift(16)
    oil_daily["oil_lag_28"] = price.shift(28)
    oil_daily["oil_lag_56"] = price.shift(56)
    oil_daily["oil_vol_28"] = daily_ret.rolling(28, min_periods=2).std()

    lag_cols = ["oil_lag_16", "oil_lag_28", "oil_lag_56"]
    oil_daily[lag_cols] = oil_daily[lag_cols].bfill()
    ret_cols = ["oil_ret_7", "oil_ret_28", "oil_vol_28"]
    oil_daily[ret_cols] = oil_daily[ret_cols].fillna(0.0)
    return oil_daily


# -------------------- Holiday date sets --------------------

def national_special_dates(holidays: pd.DataFrame) -> pd.DatetimeIndex:
    """Active National special-day dates (``transferred==False``, non-``Work Day``).

    A date qualifies when it carries a National-locale row that is active (not
    transferred) and not a work-day override. Matches the darts-family /
    :func:`store_sales.features.calendar.national_holiday_dates` definition.

    Args:
        holidays: Raw ``holidays_events.csv`` frame (``date`` parsed).

    Returns:
        Sorted, de-duplicated national special-day timestamps.
    """
    active = holidays[(holidays["transferred"] == False)  # noqa: E712
                      & (holidays["type"] != "Work Day")]
    national = active[active["locale"] == "National"]
    return pd.DatetimeIndex(pd.Series(national["date"].unique())).sort_values()


def transferred_origin_dates(holidays: pd.DataFrame) -> pd.DatetimeIndex:
    """Original dates of transferred holidays (``transferred==True``).

    These dates are worked rather than celebrated (the holiday was moved), but
    often keep a residual behavioural footprint — hence an explicit flag instead
    of silently discarding the row.

    Args:
        holidays: Raw ``holidays_events.csv`` frame (``date`` parsed).

    Returns:
        Sorted, de-duplicated transferred-origin timestamps.
    """
    origin = holidays[holidays["transferred"] == True]  # noqa: E712
    return pd.DatetimeIndex(pd.Series(origin["date"].unique())).sort_values()


# -------------------- Holiday distance / flag frames --------------------

def holiday_distance_frame(date_index, special_dates,
                           cap: int = SPECIAL_DIST_CAP) -> pd.DataFrame:
    """Continuous distance to the nearest national special day.

    Computed once on a daily calendar so the same value broadcasts to every
    ``(store, family)`` row of that date. "Days to the NEXT special day" reads
    the future calendar, which is legitimate future-covariate information (the
    holiday schedule is known in advance), not sales leakage.

    Args:
        date_index: Daily date axis to materialise the features over.
        special_dates: Dates flagged as national special days (see
            :func:`national_special_dates`).
        cap: Distances are clipped to ``[0, cap]`` (edges with no prior/next
            special day fall back to ``cap``).

    Returns:
        ``[date, days_since_prev_special, days_to_next_special]`` (``float32``).
    """
    out = (pd.DataFrame({"date": pd.DatetimeIndex(date_index)})
           .sort_values("date", ignore_index=True))
    dts = out["date"].to_numpy()
    is_spec = out["date"].isin(special_dates).to_numpy()
    spec = np.where(is_spec, dts, np.datetime64("NaT"))
    prev_spec = pd.Series(spec).ffill().to_numpy()   # last special on/before
    next_spec = pd.Series(spec).bfill().to_numpy()   # next special on/after
    one_day = np.timedelta64(1, "D")
    out["days_since_prev_special"] = (dts - prev_spec) / one_day
    out["days_to_next_special"] = (next_spec - dts) / one_day
    for c in ["days_since_prev_special", "days_to_next_special"]:
        out[c] = out[c].fillna(cap).clip(0, cap).astype(np.float32)
    return out[["date", "days_since_prev_special", "days_to_next_special"]]


def holiday_extra_frame(date_index, holidays,
                        cap: int = SPECIAL_DIST_CAP) -> pd.DataFrame:
    """Full date-level holiday-extra frame (:data:`HOLIDAY_EXTRA_COLS`).

    Convenience wrapper combining the transferred-origin flag with the
    continuous distance features, derived straight from the raw holidays frame.
    Used by the neural leg, which has the raw frame on hand.

    Args:
        date_index: Daily date axis.
        holidays: Raw ``holidays_events.csv`` frame.
        cap: Distance clip passed through to :func:`holiday_distance_frame`.

    Returns:
        ``[date, is_transferred_origin, days_to_next_special,
        days_since_prev_special]``.
    """
    out = holiday_distance_frame(date_index, national_special_dates(holidays), cap)
    origin = transferred_origin_dates(holidays)
    out["is_transferred_origin"] = out["date"].isin(origin).astype("float32")
    return out[["date", *HOLIDAY_EXTRA_COLS]]
