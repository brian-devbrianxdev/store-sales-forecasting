"""Global neural time-series forecaster (darts TSMixer / TiDE / NHiTS) on GPU.

Per-family neural training is impractical on CPU; this fits ONE global model
across all 1782 (store, family) series with covariates, producing test preds
(anchor 2017-08-16). `--model tsmixer` (with HID/FF/BLK env overrides) produces
the tuned TSMixer leg used by the champion ensemble.

Target = log1p(sales). Future covs: onpromotion, oil(ffill), calendar, holiday flags.
Past covs: transactions. Static: family/city/state/type/cluster (categorical).

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python model/neural_tsmixer.py --model tsmixer [--epochs 30]
Saves: out/submission_{model}.csv  (its standalone RMSLE = Kaggle score of that CSV)
"""
from __future__ import annotations
import argparse, time, os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
VAL_START = pd.Timestamp("2017-07-31"); TEST_START = pd.Timestamp("2017-08-16"); H = 16


def rmsle(p, t):
    p = np.clip(p, 0, None); t = np.clip(t, 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2)))


def load_frames():
    tr = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    te = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
    hol = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    sto = pd.read_csv(DATA / "stores.csv")
    return tr, te, oil, hol, sto


def holiday_flag(hol):
    """National holiday flag per date (transferred=False, not a Work Day)."""
    h = hol[(hol["transferred"] == False) & (hol["type"] != "Work Day")]
    nat = set(h[h["locale"] == "National"]["date"])
    return nat


def build_long(tr, te, oil, hol, sto):
    """Continuous-daily long df (store,family,date) with sales + covariates, train+test span."""
    full = pd.concat([tr, te.assign(sales=np.nan)], ignore_index=True, sort=False)
    keys = full[["store_nbr", "family"]].drop_duplicates()
    dmin, dmax = full["date"].min(), full["date"].max()
    cal = pd.date_range(dmin, dmax, freq="D")
    # cartesian (store,family) x date
    grid = keys.assign(k=1).merge(pd.DataFrame({"date": cal, "k": 1}), on="k").drop(columns="k")
    full = grid.merge(full, on=["store_nbr", "family", "date"], how="left")
    full["onpromotion"] = full["onpromotion"].fillna(0.0)
    full["sales"] = full["sales"]  # nan in test region kept
    # oil ffill/bfill
    oil = oil.set_index("date").reindex(cal).rename_axis("date").reset_index()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    full = full.merge(oil, on="date", how="left")
    # calendar
    full["dow"] = full["date"].dt.dayofweek.astype(float)
    full["month"] = full["date"].dt.month.astype(float)
    full["day"] = full["date"].dt.day.astype(float)
    full["payday"] = ((full["date"].dt.day == 15) | (full["date"].dt.is_month_end)).astype(float)
    nat = holiday_flag(hol)
    full["nathol"] = full["date"].isin(nat).astype(float)
    # Fourier day-of-year harmonics (yearly seasonality basis the LGBM family lacks)
    doy = full["date"].dt.dayofyear.astype(float)
    for k in (1, 2, 3):
        full[f"foy_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        full[f"foy_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    # static from stores
    full = full.merge(sto, on="store_nbr", how="left")
    full = full.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return full


def make_series(full, anchor, model_name, epochs, gpu):
    from darts import TimeSeries
    from darts.models import TiDEModel, NHiTSModel, TSMixerModel
    from darts.dataprocessing.transformers import StaticCovariatesTransformer
    import torch
    t0 = time.time()

    df = full[full["date"] < anchor + pd.Timedelta(days=H)].copy()
    df["y"] = np.log1p(df["sales"].clip(lower=0)).astype(np.float32)
    df["sid"] = df["store_nbr"].astype(str) + "_" + df["family"]
    foy = [f"foy_{t}{k}" for t in ("sin", "cos") for k in (1, 2, 3)]
    base_fc = ["onpromotion", "dcoilwtico", "dow", "month", "day", "payday", "nathol"]
    use_foy = os.environ.get("FOY", "0") == "1"
    futcov_cols = (base_fc + foy) if use_foy else base_fc
    for c in futcov_cols:
        df[c] = df[c].astype(np.float32)
    for c in ["family", "city", "state", "type"]:
        df[c] = df[c].astype(str)
    df["cluster"] = df["cluster"].astype(str); df["store_nbr_s"] = df["store_nbr"].astype(str)

    static_cols = ["family", "city", "state", "type", "cluster", "store_nbr_s"]

    fit_df = df[df["date"] < anchor].copy()
    # VECTORIZED multi-series build (C-optimized; avoids slow Python per-series loop that trips
    # the node's session watchdog).
    tgt_all = TimeSeries.from_group_dataframe(
        fit_df, group_cols="sid", time_col="date", value_cols="y",
        static_cols=static_cols, freq="D", fill_missing_dates=True, fillna_value=0.0)
    fut_all = TimeSeries.from_group_dataframe(
        df, group_cols="sid", time_col="date", value_cols=futcov_cols,
        freq="D", fill_missing_dates=True, fillna_value=0.0)
    fut_by = {ts.static_covariates["sid"].item(): ts.with_static_covariates(None).astype(np.float32)
              for ts in fut_all}
    tgt_list, futf_list, sids = [], [], []
    for ts in tgt_all:
        sid = ts.static_covariates["sid"].item()
        if len(ts) < 60 or sid not in fut_by:
            continue
        sc = ts.static_covariates.drop(columns=["sid"]).reset_index(drop=True)
        tgt_list.append(ts.with_static_covariates(sc).astype(np.float32))
        futf_list.append(fut_by[sid]); sids.append(sid)
    print(f"  built {len(tgt_list)} series in {time.time()-t0:.0f}s", flush=True)

    seed = int(os.environ.get("SEED", 0))
    use_mps = (not gpu) and os.environ.get("MPS", "0") == "1" and torch.backends.mps.is_available()
    accel = "gpu" if gpu else ("mps" if use_mps else "cpu")
    common = dict(input_chunk_length=int(os.environ.get("ICL", 90)), output_chunk_length=H, n_epochs=epochs,
                  batch_size=1024, random_state=seed,
                  pl_trainer_kwargs={"accelerator": accel,
                                     "devices": [0] if gpu else 1,
                                     "enable_progress_bar": False, "enable_model_summary": False})
    dlk = {"num_workers": 4}
    if model_name == "tide":
        model = TiDEModel(hidden_size=256, num_encoder_layers=2, num_decoder_layers=2,
                          decoder_output_dim=16, temporal_width_future=4, dropout=0.1,
                          use_static_covariates=True, **common)
    elif model_name == "tsmixer":
        hid = int(os.environ.get("HID", 128)); ff = int(os.environ.get("FF", 128))
        blk = int(os.environ.get("BLK", 8)); drp = float(os.environ.get("DROP", 0.2))
        lr = float(os.environ.get("LR", 1e-3))
        model = TSMixerModel(hidden_size=hid, ff_size=ff, num_blocks=blk, dropout=drp,
                             use_static_covariates=True,
                             optimizer_kwargs={"lr": lr},
                             lr_scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
                             lr_scheduler_kwargs={"T_max": epochs}, **common)
        print(f"  tsmixer cfg hid={hid} ff={ff} blk={blk} drop={drp} lr={lr} epochs={epochs}", flush=True)
    else:
        model = NHiTSModel(num_stacks=3, num_blocks=1, num_layers=2, layer_widths=512,
                           dropout=0.1, **common)

    # capability-aware covariate routing (NHiTS: past-only, no static; TiDE: future+static)
    if not model.supports_static_covariates:
        tgt_list = [t.with_static_covariates(None) for t in tgt_list]
    else:
        sct = StaticCovariatesTransformer(); tgt_list = sct.fit_transform(tgt_list)
    cov_kw_fit, cov_kw_pred = {}, {}
    if model.supports_future_covariates:
        cov_kw_fit["future_covariates"] = futf_list; cov_kw_pred["future_covariates"] = futf_list
    elif model.supports_past_covariates:
        cov_kw_fit["past_covariates"] = futf_list; cov_kw_pred["past_covariates"] = futf_list
    print(f"  {model_name}: static={model.supports_static_covariates} "
          f"future={model.supports_future_covariates} past={model.supports_past_covariates}", flush=True)

    tf = time.time()
    try:
        model.fit(tgt_list, verbose=False, dataloader_kwargs=dlk, **cov_kw_fit)
    except TypeError:
        model.fit(tgt_list, verbose=False, **cov_kw_fit)
    print(f"  fit {model_name} {len(tgt_list)} series in {time.time()-tf:.0f}s", flush=True)
    try:
        preds = model.predict(n=H, series=tgt_list, dataloader_kwargs=dlk, **cov_kw_pred)
    except TypeError:
        preds = model.predict(n=H, series=tgt_list, **cov_kw_pred)
    rows = []
    for sid, pr in zip(sids, preds):
        sn, fam = sid.split("_", 1)
        vals = pr.values().ravel()
        for d, v in enumerate(vals):
            rows.append({"store_nbr": int(sn), "family": fam,
                         "date": anchor + pd.Timedelta(days=d), "forecast_offset": d + 1,
                         "pred_log": float(v)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tide", choices=["tide", "nhits", "tsmixer"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()
    gpu = not args.cpu

    tr, te, oil, hol, sto = load_frames()
    full = build_long(tr, te, oil, hol, sto)
    print(f"long df {full.shape}", flush=True)

    if args.skip_test:
        return
    print("[TEST]"); tdf = make_series(full, TEST_START, args.model, args.epochs, gpu)
    sub = te.merge(tdf[["date", "store_nbr", "family", "pred_log"]], on=["date", "store_nbr", "family"], how="left")
    sub["sales"] = np.expm1(sub["pred_log"]).clip(lower=0).fillna(0.0)
    sub[["id", "sales"]].sort_values("id").to_csv(OUT / f"submission_{args.model}.csv", index=False)
    print(f"Saved submission_{args.model}.csv mean={sub['sales'].mean():.2f}")


if __name__ == "__main__":
    main()
