"""Global neural time-series forecaster (darts TSMixer / TiDE / NHiTS).

Fits ONE global model across all (store, family) series with covariates,
producing test predictions (anchor 2017-08-16). ``--model tsmixer`` reproduces
the tuned TSMixer leg used by the champion ensemble.

Target = ``log1p(sales)``. Future covs: onpromotion, oil (ffill), calendar,
holiday flag. Past covs: transactions. Static: family/city/state/type/cluster.

Hyperparameters (input chunk, hidden/ff/block sizes, dropout, lr, seed) come from
the ``neural`` section of ``config.yaml`` instead of environment variables, and
output is written straight into ``submissions/`` (no separate ``out/`` dir).

Usage:
    store-sales train tsmixer --epochs 30
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .. import paths
from ..config import get_config
from ..io.data_loading import load_raw_frames
from ..features.calendar import national_holiday_dates

_cfg = get_config()
DATA = paths.DATA
OUT = paths.SUBMISSIONS
VAL_START = _cfg.common.val_start
TEST_START = _cfg.common.test_start
H = _cfg.common.horizon


def load_frames() -> tuple[pd.DataFrame, ...]:
    """Load ``(train, test, oil, holidays, stores)`` for the neural leg."""
    frames = load_raw_frames()
    return (frames["train"], frames["test"], frames["oil"],
            frames["holidays"], frames["stores"])


def build_long(tr: pd.DataFrame, te: pd.DataFrame, oil: pd.DataFrame,
               hol: pd.DataFrame, sto: pd.DataFrame) -> pd.DataFrame:
    """Continuous-daily long frame (store, family, date) with sales + covariates.

    Spans train + test, builds the full ``(store, family) × date`` grid, attaches
    oil, calendar features, the national-holiday flag, day-of-year Fourier
    harmonics, and static store metadata.

    Args:
        tr, te, oil, hol, sto: Raw train/test/oil/holidays/stores frames.

    Returns:
        The long covariate frame, sorted by ``(store_nbr, family, date)``.
    """
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
    nat = national_holiday_dates(hol)
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


def make_series(full: pd.DataFrame, anchor: pd.Timestamp, model_name: str,
                epochs: int, gpu: bool, mps: bool = False) -> pd.DataFrame:
    """Build per-series darts inputs, fit the model, and return predictions.

    Args:
        full: The long covariate frame from :func:`build_long`.
        anchor: Forecast anchor date.
        model_name: One of ``"tide"``, ``"nhits"``, ``"tsmixer"``.
        epochs: Training epochs.
        gpu: If True, train on GPU; otherwise CPU (or MPS if ``mps``).
        mps: If True (and not ``gpu``), prefer Apple MPS when available.

    Returns:
        A long frame ``(store_nbr, family, date, forecast_offset, pred_log)``.
    """
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
    use_foy = _cfg.neural.use_foy
    futcov_cols = (base_fc + foy) if use_foy else base_fc
    for c in futcov_cols:
        df[c] = df[c].astype(np.float32)
    for c in ["family", "city", "state", "type"]:
        df[c] = df[c].astype(str)
    df["cluster"] = df["cluster"].astype(str); df["store_nbr_s"] = df["store_nbr"].astype(str)

    static_cols = ["family", "city", "state", "type", "cluster", "store_nbr_s"]

    fit_df = df[df["date"] < anchor].copy()
    # VECTORIZED multi-series build (C-optimized; avoids slow Python per-series loop).
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

    seed = _cfg.neural.seed
    use_mps = (not gpu) and mps and torch.backends.mps.is_available()
    accel = "gpu" if gpu else ("mps" if use_mps else "cpu")
    common = dict(input_chunk_length=_cfg.neural.input_chunk_length, output_chunk_length=H,
                  n_epochs=epochs, batch_size=_cfg.neural.batch_size, random_state=seed,
                  pl_trainer_kwargs={"accelerator": accel,
                                     "devices": [0] if gpu else 1,
                                     "enable_progress_bar": False, "enable_model_summary": False})
    dlk = {"num_workers": 4}
    if model_name == "tide":
        tide = _cfg.neural.tide
        model = TiDEModel(hidden_size=tide["hidden_size"],
                          num_encoder_layers=tide["num_encoder_layers"],
                          num_decoder_layers=tide["num_decoder_layers"],
                          decoder_output_dim=tide["decoder_output_dim"],
                          temporal_width_future=tide["temporal_width_future"],
                          dropout=tide["dropout"],
                          use_static_covariates=True, **common)
    elif model_name == "tsmixer":
        tsm = _cfg.neural.tsmixer
        hid, ff, blk = tsm["hidden_size"], tsm["ff_size"], tsm["num_blocks"]
        drp, lr = tsm["dropout"], tsm["lr"]
        model = TSMixerModel(hidden_size=hid, ff_size=ff, num_blocks=blk, dropout=drp,
                             use_static_covariates=True,
                             optimizer_kwargs={"lr": lr},
                             lr_scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
                             lr_scheduler_kwargs={"T_max": epochs}, **common)
        print(f"  tsmixer cfg hid={hid} ff={ff} blk={blk} drop={drp} lr={lr} epochs={epochs}", flush=True)
    else:
        nhits = _cfg.neural.nhits
        model = NHiTSModel(num_stacks=nhits["num_stacks"], num_blocks=nhits["num_blocks"],
                           num_layers=nhits["num_layers"], layer_widths=nhits["layer_widths"],
                           dropout=nhits["dropout"], **common)

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

    # Assemble the prediction frame. The original built it with a nested
    # per-(series, day) Python loop; here each series' H-day block is built
    # vectorized (offsets 1..n, dates anchor+0..n-1) and concatenated — exactly
    # the same rows, faster.
    frames = []
    for sid, pr in zip(sids, preds):
        sn, fam = sid.split("_", 1)
        vals = pr.values().ravel()
        n = len(vals)
        offsets = np.arange(n)
        frames.append(pd.DataFrame({
            "store_nbr": int(sn),
            "family": fam,
            "date": anchor + pd.to_timedelta(offsets, unit="D"),
            "forecast_offset": offsets + 1,
            "pred_log": vals.astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


def run(model_name: str, epochs: int, gpu: bool, mps: bool = False,
        out_name: str | None = None) -> None:
    """Train one neural model and write its submission CSV into ``submissions/``.

    Args:
        model_name: ``tide`` | ``nhits`` | ``tsmixer``.
        epochs: Training epochs.
        gpu: Train on GPU if True.
        mps: Prefer Apple MPS when not on GPU.
        out_name: Output filename; defaults to ``submission_{model}.csv``.
    """
    if out_name is None:
        out_name = _cfg.neural.out_name_template.format(model=model_name)
    paths.ensure_dirs()

    tr, te, oil, hol, sto = load_frames()
    full = build_long(tr, te, oil, hol, sto)
    print(f"long df {full.shape}", flush=True)

    print("[TEST]")
    tdf = make_series(full, TEST_START, model_name, epochs, gpu, mps)
    sub = te.merge(tdf[["date", "store_nbr", "family", "pred_log"]],
                   on=["date", "store_nbr", "family"], how="left")
    sub["sales"] = np.expm1(sub["pred_log"]).clip(lower=0).fillna(0.0)
    sub[["id", "sales"]].sort_values("id").to_csv(OUT / out_name, index=False)
    print(f"Saved {out_name} mean={sub['sales'].mean():.2f}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the neural leg."""
    ap = argparse.ArgumentParser(description="Global neural forecaster (darts)")
    ap.add_argument("--model", default=_cfg.neural.default_model,
                     choices=["tide", "nhits", "tsmixer"])
    ap.add_argument("--epochs", type=int, default=_cfg.neural.default_epochs)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mps", action="store_true", help="prefer Apple MPS when not on GPU")
    ap.add_argument("--out-name", default=None,
                     help="output filename (default submission_{model}.csv)")
    args = ap.parse_args(argv)
    run(args.model, args.epochs, gpu=not args.cpu, mps=args.mps, out_name=args.out_name)


if __name__ == "__main__":
    main()
