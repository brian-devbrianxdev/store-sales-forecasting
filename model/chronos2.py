"""Chronos-2 (Amazon, 2025 SOTA) on all 1782 series — newest foundation family.

MUST run in isolated env: .venv_chronos2/bin/python (py3.11 + chronos-forecasting 2.x).
Main py3.9 venv has chronos 1.5.3 (no Chronos2Pipeline). Reads shared HF cache.

Saves:
  submissions/submission_chronos2.csv
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
VAL_START = pd.Timestamp("2017-07-31")
TEST_START = pd.Timestamp("2017-08-16")
H = 16
CTX = 512


def rmsle(p, t):
    p = np.clip(p, 0, None); t = np.clip(t, 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2)))


def run_pass(pipe, train, anchor, batch_size):
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
    rows = []
    for sf, qi in zip(sf_list, q):
        med = np.asarray(qi)[0, :, 0]  # (variates=1, H, levels=1) -> (H,) 0.5 quantile
        sn, fam = sf
        for d, val in enumerate(med):
            rows.append({"store_nbr": sn, "family": fam,
                         "date": anchor + pd.Timedelta(days=d), "forecast_offset": d + 1,
                         "pred_raw": float(np.clip(val, 0, None))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

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
