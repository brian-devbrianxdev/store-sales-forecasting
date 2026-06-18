"""Chronos-2 conditioned on covariates — improve the champion's keeper expert.

The plain Chronos-2 leg runs zero-shot on the raw sales context only
(:mod:`store_sales.models.chronos2`). Chronos-2 v2.x supports covariates via
``predict_df``: columns in ``df`` that are not the target are past covariates;
columns also present in ``future_df`` become known-future covariates.
``onpromotion`` is known for the full horizon → ideal known-future covariate;
oil is past-only; the holiday flag is known-future.

MUST run in ``.venv_chronos2/bin/python`` (Python 3.11 + chronos-forecasting 2.x).
Saves ``submissions/submission_chronos2_cov{suffix}.csv``.

Note: :func:`build_holiday_flag` here is an **any-locale** active-special-day
count, intentionally distinct from the national-only
:func:`store_sales.features.calendar.national_holiday_dates` used by the other
legs; it is kept local to preserve this leg's exact behaviour.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .. import paths
from ..config import get_config

_cfg = get_config()
DATA = paths.DATA
OUT = paths.SUBMISSIONS
VAL_START = _cfg.common.val_start
TEST_START = _cfg.common.test_start
H = _cfg.common.horizon
CTX = _cfg.common.context_len


def build_holiday_flag() -> pd.Series:
    """National-level special-day flag per date (simple, locale-agnostic).

    1 if an active (non-transferred-away) Holiday/Additional/Bridge/Event/Transfer
    day across any locale.

    Returns:
        A 0/1 ``float32`` Series indexed by date.
    """
    hol = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    active = hol[(hol["transferred"] == False) & (hol["type"] != "Work Day")]  # noqa: E712
    flag = active.groupby("date").size().clip(upper=1).astype(np.float32)
    return flag  # Series indexed by date


def build_oil_daily() -> pd.Series:
    """Daily oil price (ffill/bfill) indexed by date through 2017-09-01."""
    oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"]).set_index("date")["dcoilwtico"]
    full = pd.date_range(oil.index.min(), pd.Timestamp("2017-09-01"), freq="D")
    return oil.reindex(full).ffill().bfill().astype(np.float32)  # Series indexed by date


def make_panel(train: pd.DataFrame, test: pd.DataFrame, use_oil: bool,
               use_holiday: bool) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Long panel of all series with continuous timestamps, target + covariates."""
    hol_flag = build_holiday_flag() if use_holiday else None
    oil_daily = build_oil_daily() if use_oil else None

    # union of train + test rows (onpromotion known on both)
    tr = train[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    te = test[["date", "store_nbr", "family", "onpromotion"]].copy()
    te["sales"] = np.nan
    panel = pd.concat([tr, te], ignore_index=True)
    panel["item_id"] = panel["store_nbr"].astype(str) + "__" + panel["family"]
    return panel, hol_flag, oil_daily


def reindex_series(g: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Continuous daily reindex of one series between ``[lo, hi]`` inclusive."""
    idx = pd.date_range(lo, hi, freq="D")
    g = g.set_index("date").reindex(idx)
    g["sales"] = g["sales"].fillna(0.0)
    g["onpromotion"] = g["onpromotion"].fillna(0.0)
    return g


def run_pass(pipe, panel, hol_flag, oil_daily, anchor, batch_size, use_oil, use_holiday):
    """Build df (context) + future_df (horizon) per series and call ``predict_df``.

    The per-series Python loop is intrinsic here (each series needs its own
    context/future frame with covariates), so it is preserved as-is.
    """
    hist_lo = anchor - pd.Timedelta(days=CTX)
    fut_hi = anchor + pd.Timedelta(days=H - 1)
    df_rows, fut_rows = [], []
    items = panel.groupby("item_id")
    t0 = time.time()
    for item_id, g in items:
        g = g.sort_values("date")
        store = g["store_nbr"].iloc[0]; fam = g["family"].iloc[0]
        # context window
        ctx = g[(g["date"] >= hist_lo) & (g["date"] < anchor)]
        if ctx.empty:
            continue
        ctx = reindex_series(ctx[["date", "sales", "onpromotion"]], ctx["date"].min(), anchor - pd.Timedelta(days=1))
        ctx = ctx.reset_index().rename(columns={"index": "timestamp"})
        rec = {"item_id": item_id, "timestamp": ctx["timestamp"], "target": ctx["sales"].astype(np.float32),
               "onpromotion": ctx["onpromotion"].astype(np.float32)}
        if use_oil:
            rec["oil"] = oil_daily.reindex(ctx["timestamp"]).ffill().bfill().values.astype(np.float32)
        if use_holiday:
            rec["hol"] = ctx["timestamp"].map(lambda d: float(hol_flag.get(d, 0.0))).astype(np.float32).values
        df_rows.append(pd.DataFrame(rec))
        # horizon future covariates (onpromotion known)
        fut = g[(g["date"] >= anchor) & (g["date"] <= fut_hi)][["date", "onpromotion"]]
        fut = fut.set_index("date").reindex(pd.date_range(anchor, fut_hi, freq="D"))
        fut["onpromotion"] = fut["onpromotion"].fillna(0.0)
        frec = {"item_id": item_id, "timestamp": fut.index,
                "onpromotion": fut["onpromotion"].astype(np.float32).values}
        if use_holiday:
            frec["hol"] = [float(hol_flag.get(d, 0.0)) for d in fut.index]
        fut_rows.append(pd.DataFrame(frec))
    df = pd.concat(df_rows, ignore_index=True)
    future_df = pd.concat(fut_rows, ignore_index=True)
    print(f"  panel built ({len(df_rows)} series) in {time.time()-t0:.0f}s; predicting...", flush=True)
    t0 = time.time()
    out = pipe.predict_df(df, future_df=future_df, id_column="item_id", timestamp_column="timestamp",
                          target="target", prediction_length=H, quantile_levels=[0.5],
                          batch_size=batch_size, validate_inputs=False)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    # out columns: item_id, timestamp, target_name, predictions, "0.5"
    qcol = "0.5" if "0.5" in out.columns else "predictions"
    out = out.rename(columns={qcol: "pred_raw"})
    out["forecast_offset"] = out.groupby("item_id").cumcount() + 1
    out["store_nbr"] = out["item_id"].str.split("__").str[0].astype(int)
    out["family"] = out["item_id"].str.split("__", n=1).str[1]
    out = out.rename(columns={"timestamp": "date"})
    out["pred_raw"] = out["pred_raw"].clip(lower=0)
    return out[["date", "store_nbr", "family", "forecast_offset", "pred_raw"]]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the covariate-conditioned Chronos-2 leg."""
    from chronos import Chronos2Pipeline

    ap = argparse.ArgumentParser(description="Chronos-2 with covariates leg")
    ap.add_argument("--model", default=_cfg.chronos.model)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--batch-size", type=int, default=_cfg.chronos.cov_batch_size)
    ap.add_argument("--oil", action="store_true", help="add oil as past-only covariate")
    ap.add_argument("--holiday", action="store_true", help="add national holiday flag (known-future)")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)

    print(f"Loading {args.model} ... (oil={args.oil} holiday={args.holiday})")
    try:
        pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cpu")
    except TypeError:
        pipe = Chronos2Pipeline.from_pretrained(args.model)
    print("OK")

    train = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    panel, hol_flag, oil_daily = make_panel(train, test, args.oil, args.holiday)

    if args.skip_test:
        return
    print("\n[TEST] window 2017-08-16..08-31")
    test_df = run_pass(pipe, panel, hol_flag, oil_daily, TEST_START, args.batch_size, args.oil, args.holiday)
    sub = test.merge(test_df[["date", "store_nbr", "family", "pred_raw"]],
                     on=["date", "store_nbr", "family"], how="left")
    sub["sales"] = sub["pred_raw"].clip(lower=0).fillna(0)
    out = sub[["id", "sales"]].sort_values("id")
    out.to_csv(OUT / f"submission_chronos2_cov{args.suffix}.csv", index=False)
    print(f"Saved submission_chronos2_cov{args.suffix}.csv: rows={len(out)} mean={out['sales'].mean():.2f}")


if __name__ == "__main__":
    main()
