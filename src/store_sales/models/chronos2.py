"""Chronos-2 (Amazon, 2025) zero-shot on all 1782 series.

MUST run in the isolated env ``.venv_chronos2/bin/python`` (Python 3.11 +
``chronos-forecasting`` 2.x); the main Python 3.9 venv ships chronos 1.5.3, which
lacks ``Chronos2Pipeline``. Reads the shared Hugging Face cache.

Saves ``submissions/submission_chronos2{suffix}.csv``.

Usage:
    .venv_chronos2/bin/python -m store_sales.models.chronos2
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


def run_pass(pipe, train: pd.DataFrame, anchor: pd.Timestamp,
             batch_size: int) -> pd.DataFrame:
    """Predict the horizon for every (store, family) series from its raw context.

    Args:
        pipe: A loaded ``Chronos2Pipeline``.
        train: Raw training frame (``date`` parsed).
        anchor: Forecast anchor date.
        batch_size: Prediction batch size.

    Returns:
        A long frame ``(store_nbr, family, date, forecast_offset, pred_raw)``.
    """
    series = train[train["date"] < anchor].sort_values(["store_nbr", "family", "date"])
    keys = series.groupby(["store_nbr", "family"]).indices
    sf_list = list(keys.keys())
    sales = series["sales"].values.astype(np.float32)
    contexts = [sales[keys[sf]][-CTX:] for sf in sf_list]
    print(f"  {len(sf_list)} series; predicting...", flush=True)
    t0 = time.time()
    q, _mean = pipe.predict_quantiles(contexts, prediction_length=H,
                                      quantile_levels=[0.5], batch_size=batch_size)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    # Assemble per-series blocks (vectorized over the H horizon days) and concat.
    # Equivalent to the original nested per-(series, day) loop.
    frames = []
    for sf, qi in zip(sf_list, q):
        med = np.asarray(qi)[0, :, 0]  # (variates=1, H, levels=1) -> (H,) 0.5 quantile
        sn, fam = sf
        n = len(med)
        offsets = np.arange(n)
        frames.append(pd.DataFrame({
            "store_nbr": sn,
            "family": fam,
            "date": anchor + pd.to_timedelta(offsets, unit="D"),
            "forecast_offset": offsets + 1,
            "pred_raw": np.clip(med, 0, None).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the zero-shot Chronos-2 leg."""
    from chronos import Chronos2Pipeline

    ap = argparse.ArgumentParser(description="Chronos-2 zero-shot leg")
    ap.add_argument("--model", default=_cfg.chronos.model)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--batch-size", type=int, default=_cfg.chronos.plain_batch_size)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)

    print(f"Loading {args.model} ...")
    try:
        pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cpu")
    except TypeError:
        pipe = Chronos2Pipeline.from_pretrained(args.model)
    print("OK")

    train = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date"])

    if args.skip_test:
        return
    print("\n[TEST] window 2017-08-16..08-31")
    test_df = run_pass(pipe, train, TEST_START, args.batch_size)
    sub = test.merge(test_df[["date", "store_nbr", "family", "pred_raw"]],
                     on=["date", "store_nbr", "family"], how="left")
    sub["sales"] = sub["pred_raw"].clip(lower=0).fillna(0)
    out = sub[["id", "sales"]].sort_values("id")
    out.to_csv(OUT / f"submission_chronos2{args.suffix}.csv", index=False)
    print(f"Saved submission_chronos2{args.suffix}.csv: rows={len(out)} mean={out['sales'].mean():.2f}")


if __name__ == "__main__":
    main()
