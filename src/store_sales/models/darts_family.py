"""Per-family GBT family members (darts-based).

Recipe: per-family ensemble of 4 lag configs (lags = 56, 7, 365, 730), trained
on full data AND a 2015+ subset, then averaged. Each *variant* (base, deeper,
xgb, subsampled, weighted, cat_deep) selects a different model/regularisation;
together they form the ``family`` sub-blend in
:mod:`store_sales.ensemble.build`.

The legacy script was driven by ~15 environment variables; those are now the
``darts_family`` section of ``config.yaml``. A variant is resolved into a typed
:class:`DartsSettings` and threaded through the pipeline, so behaviour is
identical but fully configuration-driven.

Usage:
    store-sales train darts-family --variant base
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from .. import paths
from ..config import get_config
from ..io.data_loading import load_raw_frames

warnings.filterwarnings("ignore")

from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    InvertibleMapper,
    Scaler,
    StaticCovariatesTransformer,
)
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from darts.models import CatBoostModel, LightGBMModel, XGBModel
from darts.models.filtering.moving_average_filter import MovingAverageFilter

_cfg = get_config()
PATH = paths.DATA

FORECAST_HORIZON = _cfg.darts_family.forecast_horizon
ZERO_FC_WINDOW = _cfg.darts_family.zero_fc_window
SELECTED_HOLIDAYS = list(_cfg.darts_family.selected_holidays)
# Andes/Amazon provinces on the Sierra school calendar (year starts ~Sep →
# August back-to-school surge). Coast provinces start ~May (no August surge).
SIERRA_STATES = set(_cfg.darts_family.sierra_states)


@dataclass(frozen=True)
class DartsSettings:
    """Resolved settings for one darts-family variant (replaces the env vars)."""

    seed: int
    depth: int
    n_estimators: int
    lr: float
    include_xgb: bool
    cat_only: bool
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    objective: str
    tweedie_variance_power: float
    log_transform: bool
    floor_feature: bool
    sample_weight_floor: bool
    new_feats: bool
    new_feats_famtotal: bool
    weight_scheme: str
    school_region: bool
    lags_main: int
    lags_extra: list[int]
    lags_past_covariates: list[int]
    lags_future_covariates: tuple[int, int]
    output_chunk_length: int
    out_name: str

    @classmethod
    def from_variant(cls, name: str) -> "DartsSettings":
        """Resolve a named variant from config into typed settings."""
        merged = _cfg.darts_family.variant(name)
        return cls(
            seed=int(merged["seed"]),
            depth=int(merged["depth"]),
            n_estimators=int(merged["n_estimators"]),
            lr=float(merged["lr"]),
            include_xgb=bool(merged["include_xgb"]),
            cat_only=bool(merged["cat_only"]),
            feature_fraction=float(merged["feature_fraction"]),
            bagging_fraction=float(merged["bagging_fraction"]),
            bagging_freq=int(merged["bagging_freq"]),
            objective=str(merged["objective"]),
            tweedie_variance_power=float(merged["tweedie_variance_power"]),
            log_transform=bool(merged["log_transform"]),
            floor_feature=bool(merged["floor_feature"]),
            sample_weight_floor=bool(merged["sample_weight_floor"]),
            new_feats=bool(merged["new_feats"]),
            new_feats_famtotal=bool(merged["new_feats_famtotal"]),
            weight_scheme=str(merged["weight_scheme"]),
            school_region=bool(merged["school_region"]),
            lags_main=int(merged["lags_main"]),
            lags_extra=list(merged["lags_extra"]),
            lags_past_covariates=list(merged["lags_past_covariates"]),
            lags_future_covariates=tuple(merged["lags_future_covariates"]),
            output_chunk_length=int(merged["output_chunk_length"]),
            out_name=str(merged["out_name"]),
        )


# -------------------- Data loading & preprocessing --------------------

def load_data() -> tuple[pd.DataFrame, ...]:
    """Load raw frames in the order the darts pipeline expects.

    Returns:
        ``(train, test, oil, store, transaction, holiday)`` where ``oil`` has its
        ``dcoilwtico`` column renamed to ``oil``.
    """
    frames = load_raw_frames()
    train = frames["train"]
    test = frames["test"]
    oil = frames["oil"].rename(columns={"dcoilwtico": "oil"})
    store = frames["stores"]
    transaction = frames["transactions"]
    holiday = frames["holidays"]
    return train, test, oil, store, transaction, holiday


def reindex_train(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Reindex train onto a complete ``(date × store × family)`` grid."""
    train_start = train.date.min()
    train_end = train.date.max()
    multi_idx = pd.MultiIndex.from_product(
        [
            pd.date_range(train_start, train_end),
            train.store_nbr.unique(),
            train.family.unique(),
        ],
        names=["date", "store_nbr", "family"],
    )
    train = train.set_index(["date", "store_nbr", "family"]).reindex(multi_idx).reset_index()
    train[["sales", "onpromotion"]] = train[["sales", "onpromotion"]].fillna(0.0)
    train.id = train.id.interpolate(method="linear")
    return train, train_start, train_end


def fill_oil(oil: pd.DataFrame, train_start: pd.Timestamp,
             test_end: pd.Timestamp) -> pd.DataFrame:
    """Daily oil series across the full span, linearly interpolated."""
    oil = (
        oil.merge(
            pd.DataFrame({"date": pd.date_range(train_start, test_end)}),
            on="date",
            how="outer",
        )
        .sort_values("date", ignore_index=True)
    )
    oil.oil = oil.oil.interpolate(method="linear", limit_direction="both")
    return oil


def fill_transactions(transaction: pd.DataFrame, train: pd.DataFrame,
                      num_store: int) -> pd.DataFrame:
    """Fill the transactions series, zeroing closed-store days, then interpolating."""
    store_sales = train.groupby(["date", "store_nbr"]).sales.sum().reset_index()
    transaction = transaction.merge(
        store_sales, on=["date", "store_nbr"], how="outer"
    ).sort_values(["date", "store_nbr"], ignore_index=True)
    transaction.loc[transaction.sales.eq(0), "transactions"] = 0.0
    transaction = transaction.drop(columns=["sales"])
    transaction.transactions = transaction.groupby("store_nbr", group_keys=False).transactions.apply(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    return transaction


def process_holidays(holiday: pd.DataFrame,
                     store: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean holiday descriptions and split into work-day and national tables."""
    def _process_holiday(s):
        if "futbol" in s:
            return "futbol"
        to_remove = list(set(store.city.str.lower()) | set(store.state.str.lower()))
        for w in to_remove:
            s = s.replace(w, "")
        return s

    holiday.description = (
        holiday.apply(
            lambda x: x.description.lower().replace(x.locale_name.lower(), ""),
            axis=1,
        )
        .apply(_process_holiday)
        .replace(r"[+-]\d+|\b(de|del|traslado|recupero|puente|-)\b", "", regex=True)
        .replace(r"\s+|-", " ", regex=True)
        .str.strip()
    )

    holiday = holiday[holiday.transferred.eq(False)]

    work_days = holiday[holiday.type.eq("Work Day")][["date", "type"]].rename(
        columns={"type": "work_day"}
    ).reset_index(drop=True)
    work_days.work_day = work_days.work_day.notna().astype(int)

    holiday = holiday[holiday.type != "Work Day"].reset_index(drop=True)

    national_holidays = holiday[holiday.locale.eq("National")][["date", "description"]].reset_index(drop=True)
    national_holidays = national_holidays[~national_holidays.duplicated()]
    national_holidays = pd.get_dummies(national_holidays, columns=["description"], prefix="nat")
    national_holidays = national_holidays.groupby("date").sum().reset_index()
    national_holidays = national_holidays.rename(columns={"nat_primer grito independencia": "nat_primer grito"})

    return work_days, national_holidays


# -------------------- Build combined data frame --------------------

def build_data(train, test, transaction, oil, store, work_days, national_holidays,
               train_end, s: DartsSettings, anchor=None) -> pd.DataFrame:
    """Merge everything into one long frame and engineer the variant's covariates.

    Args:
        train, test, transaction, oil, store, work_days, national_holidays:
            Preprocessed component frames.
        train_end: Last training date.
        s: Resolved variant settings (gates the optional feature blocks).
        anchor: Forecast anchor used to leak-proof the ``fam_total`` future cov.

    Returns:
        The combined long dataframe with engineered covariate columns.
    """
    keep_national_holidays = national_holidays[["date", *SELECTED_HOLIDAYS]]
    data = (
        pd.concat([train, test], axis=0, ignore_index=True)
        .merge(transaction, on=["date", "store_nbr"], how="left")
        .merge(oil, on="date", how="left")
        .merge(store, on="store_nbr", how="left")
        .merge(work_days, on="date", how="left")
        .merge(keep_national_holidays, on="date", how="left")
        .sort_values(["date", "store_nbr", "family"], ignore_index=True)
    )
    data[["work_day", *SELECTED_HOLIDAYS]] = data[["work_day", *SELECTED_HOLIDAYS]].fillna(0)

    if s.floor_feature or s.sample_weight_floor:
        from ..features.floor import ensure_floor_parquet
        floor_path = ensure_floor_parquet(train, PATH)
        floor = pd.read_parquet(floor_path)[
            ["date", "store_nbr", "family", "cell_floor"]
        ]
        data = data.merge(floor, on=["date", "store_nbr", "family"], how="left")
        data["cell_floor"] = data["cell_floor"].fillna(data["cell_floor"].median())

    data["day"] = data.date.dt.day
    data["month"] = data.date.dt.month
    data["year"] = data.date.dt.year
    data["day_of_week"] = data.date.dt.dayofweek
    data["day_of_year"] = data.date.dt.dayofyear
    data["week_of_year"] = data.date.dt.isocalendar().week.astype(int)
    data["date_index"] = data.date.factorize()[0]

    if s.school_region:
        sierra = data.state.isin(SIERRA_STATES).to_numpy(dtype=np.float32)
        doy = data.date.dt.dayofyear.to_numpy()
        # triangular bump peaking at Sep-1 (doy≈244), ±45-day support.
        ramp = np.clip(1.0 - np.abs(doy - 244) / 45.0, 0.0, None).astype(np.float32)
        data["sierra_school_ramp"] = sierra * ramp
        in_late_summer = data.date.dt.month.isin([8, 9]).to_numpy(dtype=np.float32)
        data["sierra_late_summer"] = sierra * in_late_summer

    if s.new_feats:
        dom = data.date.dt.day
        eom = data.date.dt.daysinmonth
        # explicit payday shape (Ecuador public sector paid 15th + month-end)
        data["days_to_15"] = (dom - 15).abs().astype(np.float32)
        data["days_to_eom"] = (eom - dom).astype(np.float32)
        data["is_payday"] = ((dom == 15) | (dom == eom)).astype(np.float32)
        # extended 2016 earthquake aftermath (decaying exogenous shock).
        dq = (data.date - pd.Timestamp("2016-04-16")).dt.days
        data["quake_after"] = np.where((dq >= 0) & (dq <= 45), np.exp(-dq / 21.0), 0.0).astype(np.float32)
        # cross-series national family demand, log1p, leak-proofed by masking
        # sales strictly after the anchor before the sum, then ffilling.
        if s.new_feats_famtotal:
            s_agg = data["sales"].where(data["date"] <= anchor) if anchor is not None else data["sales"]
            ft = data.assign(_s=s_agg).groupby(["date", "family"])["_s"].sum(min_count=1).rename("fam_total").reset_index()
            ft = ft.sort_values(["family", "date"])
            ft["fam_total"] = ft.groupby("family")["fam_total"].ffill()
            data = data.merge(ft, on=["date", "family"], how="left")
            data["fam_total"] = np.log1p(data["fam_total"].fillna(0).clip(lower=0)).astype(np.float32)

    train_start = train.date.min()
    missing_dates = pd.date_range(train_start, train_end).difference(train.date.unique()).strftime("%Y-%m-%d").tolist()
    zero_sales_dates = missing_dates + [f"{j}-01-01" for j in range(2013, 2018)]
    data.loc[
        (data.date.isin(zero_sales_dates)) & (data.sales.eq(0)) & (data.onpromotion.eq(0)),
        ["sales", "onpromotion"],
    ] = np.nan

    data.store_nbr = data.store_nbr.apply(lambda x: f"store_nbr_{x}")
    data.cluster = data.cluster.apply(lambda x: f"cluster_{x}")
    data.type = data.type.apply(lambda x: f"type_{x}")
    data.city = data.city.apply(lambda x: f"city_{x.lower()}")
    data.state = data.state.apply(lambda x: f"state_{x.lower()}")

    return data


# -------------------- Darts pipeline helpers --------------------

def get_pipeline(static_covs_transform: bool = False,
                 log_transform: bool = False) -> Pipeline:
    """Build a darts preprocessing pipeline (fill → [static] → [log] → scale)."""
    lst = [MissingValuesFiller(n_jobs=-1)]
    if static_covs_transform:
        lst.append(StaticCovariatesTransformer(transformer_cat=OneHotEncoder(), n_jobs=-1))
    if log_transform:
        lst.append(InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, n_jobs=-1))
    lst.append(Scaler())
    return Pipeline(lst)


def get_target_series(data, static_cols, train_end, log_transform=True):
    """Per-family target TimeSeries dict, fitted preprocessing pipeline, and ids."""
    target_dict, pipe_dict, id_dict = {}, {}, {}
    for fam in tqdm(data.family.unique(), desc="Targets"):
        df = data[(data.family.eq(fam)) & (data.date.le(train_end.strftime("%Y-%m-%d")))]
        pipe = get_pipeline(True, log_transform=log_transform)
        target = TimeSeries.from_group_dataframe(
            df=df, time_col="date", value_cols="sales",
            group_cols="store_nbr", static_cols=static_cols,
        )
        target_id = [{"store_nbr": t.static_covariates.store_nbr[0], "family": fam} for t in target]
        id_dict[fam] = target_id
        target = pipe.fit_transform(target)
        target_dict[fam] = [t.astype(np.float32) for t in target]
        pipe_dict[fam] = pipe[2:]
    return target_dict, pipe_dict, id_dict


def get_weight_series(data, train_end, s: DartsSettings):
    """Per-(store, family) sample_weight TimeSeries from ``cell_floor``.

    Scheme set by ``s.weight_scheme``: ``sqrt`` (default) → ``1/sqrt(cf+0.1)``,
    ``linear`` → ``1/(cf+0.1)``, ``sq`` → ``1/(cf**2+0.1)`` — each clipped to
    ``[0.1, 10]``.
    """
    weight_dict = {}
    for fam in tqdm(data.family.unique(), desc=f"Weights({s.weight_scheme})"):
        df = data[(data.family.eq(fam)) & (data.date.le(train_end.strftime("%Y-%m-%d")))].copy()
        cf = df["cell_floor"].astype(float)
        if s.weight_scheme == "linear":
            df["weight"] = (1.0 / (cf + 0.1)).clip(0.1, 10.0)
        elif s.weight_scheme == "sq":
            df["weight"] = (1.0 / (cf ** 2 + 0.1)).clip(0.1, 10.0)
        else:  # sqrt
            df["weight"] = (1.0 / np.sqrt(cf + 0.1)).clip(0.1, 10.0)
        weights = TimeSeries.from_group_dataframe(
            df=df, time_col="date", value_cols="weight",
            group_cols="store_nbr",
        )
        weights = [w.with_static_covariates(None).astype(np.float32) for w in weights]
        weight_dict[fam] = weights
    return weight_dict


def get_covariates(data, past_cols, future_cols, train_end,
                   past_ma_cols=None, future_ma_cols=None,
                   past_window_sizes=(7, 28), future_window_sizes=(7, 28)):
    """Per-family past/future covariate TimeSeries dicts (with optional MA stacks)."""
    past_dict, future_dict = {}, {}
    covs_pipe = get_pipeline()
    for fam in tqdm(data.family.unique(), desc="Covariates"):
        df = data[data.family.eq(fam)]

        past_covs = TimeSeries.from_group_dataframe(
            df=df[df.date.le(train_end.strftime("%Y-%m-%d"))],
            time_col="date", value_cols=past_cols, group_cols="store_nbr",
        )
        past_covs = [p.with_static_covariates(None) for p in past_covs]
        past_covs = covs_pipe.fit_transform(past_covs)
        if past_ma_cols is not None:
            for size in past_window_sizes:
                ma = MovingAverageFilter(window=size)
                old = [f"rolling_mean_{size}_{c}" for c in past_ma_cols]
                new = [f"{c}_ma{size}" for c in past_ma_cols]
                past_ma = [ma.filter(p[past_ma_cols]).with_columns_renamed(old, new) for p in past_covs]
                past_covs = [p.stack(pm) for p, pm in zip(past_covs, past_ma)]
        past_dict[fam] = [p.astype(np.float32) for p in past_covs]

        future_covs = TimeSeries.from_group_dataframe(
            df=df, time_col="date", value_cols=future_cols, group_cols="store_nbr",
        )
        future_covs = [f.with_static_covariates(None) for f in future_covs]
        future_covs = covs_pipe.fit_transform(future_covs)
        if future_ma_cols is not None:
            for size in future_window_sizes:
                ma = MovingAverageFilter(window=size)
                old = [f"rolling_mean_{size}_{c}" for c in future_ma_cols]
                new = [f"{c}_ma{size}" for c in future_ma_cols]
                future_ma = [ma.filter(f[future_ma_cols]).with_columns_renamed(old, new) for f in future_covs]
                future_covs = [f.stack(fm) for f, fm in zip(future_covs, future_ma)]
        future_dict[fam] = [f.astype(np.float32) for f in future_covs]
    return past_dict, future_dict


# -------------------- Trainer --------------------

def clip_nonneg(arr: np.ndarray) -> np.ndarray:
    """Clip an array to be non-negative."""
    return np.clip(arr, a_min=0.0, a_max=None)


def get_models(configs: list[dict]) -> list[Any]:
    """Instantiate a darts model per config dict (``_cls`` selects the class)."""
    out = []
    for cfg in configs:
        cfg = dict(cfg)
        cls = cfg.pop("_cls", LightGBMModel)
        out.append(cls(**cfg))
    return out


def generate_forecasts(models, train, pipe, past_covs, future_covs,
                       drop_before=None, weights=None):
    """Fit each model, forecast the horizon, zero out dormant series, and average."""
    if drop_before is not None:
        date = pd.Timestamp(drop_before) - pd.Timedelta(days=1)
        train = [t.drop_before(date) for t in train]
        if weights is not None:
            weights = [w.drop_before(date) for w in weights]
    inputs = {"series": train, "past_covariates": past_covs, "future_covariates": future_covs}
    fit_inputs = dict(inputs)
    if weights is not None:
        fit_inputs["sample_weight"] = weights

    zero_pred_df = pd.DataFrame({
        "date": pd.date_range(train[0].end_time(), periods=FORECAST_HORIZON + 1)[1:],
        "sales": np.zeros(FORECAST_HORIZON),
    })
    zero_pred = TimeSeries.from_dataframe(zero_pred_df, time_col="date", value_cols="sales")

    ens_pred = [0 for _ in range(len(train))]
    for m in models:
        m.fit(**fit_inputs)
        pred = m.predict(n=FORECAST_HORIZON, **inputs)
        pred = pipe.inverse_transform(pred)
        for j in range(len(train)):
            if train[j][-ZERO_FC_WINDOW:].values().sum() == 0:
                pred[j] = zero_pred
        pred = [p.map(clip_nonneg) for p in pred]
        for j in range(len(ens_pred)):
            ens_pred[j] = ens_pred[j] + pred[j] / len(models)
    return ens_pred


def ensemble_predict(target_dict, pipe_dict, id_dict, past_dict, future_dict,
                     configs, drop_before=None, weight_dict=None):
    """Run :func:`generate_forecasts` per family and concatenate the results."""
    forecasts = []
    for fam in tqdm(target_dict.keys(), desc=f"Predict (drop_before={drop_before})"):
        target = target_dict[fam]
        pipe = pipe_dict[fam]
        target_id = id_dict[fam]
        past_covs = past_dict[fam]
        future_covs = future_dict[fam]
        weights = weight_dict[fam] if weight_dict is not None else None
        models = get_models(configs)
        ens_pred = generate_forecasts(models, target, pipe, past_covs, future_covs, drop_before, weights=weights)
        ens_pred = [p.to_dataframe().assign(**i) for p, i in zip(ens_pred, target_id)]
        forecasts.append(pd.concat(ens_pred, axis=0))
    forecasts = pd.concat(forecasts, axis=0).rename_axis(None, axis=1).reset_index(names="date")
    return forecasts


def build_configs(s: DartsSettings) -> list[dict]:
    """Build the list of per-model config dicts from the variant settings.

    Mirrors the original env-driven config assembly: a base config replicated
    across the lag set, optional CatBoost swap / objective / LightGBM
    subsample-feature extras, and an optional XGBoost block.
    """
    lgbm_extra: dict[str, Any] = {}
    if s.feature_fraction < 1.0:
        lgbm_extra["feature_fraction"] = s.feature_fraction
    if s.bagging_fraction < 1.0 and s.bagging_freq > 0:
        lgbm_extra["bagging_fraction"] = s.bagging_fraction
        lgbm_extra["bagging_freq"] = s.bagging_freq
    base_config: dict[str, Any] = {
        "random_state": s.seed,
        "lags": s.lags_main,
        "lags_past_covariates": list(s.lags_past_covariates),
        "lags_future_covariates": s.lags_future_covariates,
        "output_chunk_length": s.output_chunk_length,
        "n_estimators": s.n_estimators,
        "learning_rate": s.lr,
        "max_depth": s.depth,
    }
    if s.objective and not s.cat_only:
        base_config["objective"] = s.objective
        if s.objective == "tweedie":
            base_config["tweedie_variance_power"] = s.tweedie_variance_power
    if s.cat_only:
        base_config["_cls"] = CatBoostModel
        base_config["verbose"] = False
    else:
        base_config.update(lgbm_extra)
    configs = [base_config] + [{**base_config, "lags": lag} for lag in s.lags_extra]
    if s.include_xgb:
        xgb_base: dict[str, Any] = {
            "random_state": s.seed,
            "lags": s.lags_main,
            "lags_past_covariates": list(s.lags_past_covariates),
            "lags_future_covariates": s.lags_future_covariates,
            "output_chunk_length": s.output_chunk_length,
            "n_estimators": s.n_estimators,
            "learning_rate": s.lr,
            "max_depth": s.depth,
            "tree_method": "hist",
            "_cls": XGBModel,
        }
        configs += [xgb_base] + [{**xgb_base, "lags": lag} for lag in s.lags_extra]
    return configs


# -------------------- Main --------------------

def run(s: DartsSettings) -> None:
    """Train one darts-family variant and write its submission CSV."""
    out_path = paths.SUBMISSIONS / s.out_name

    print(">>> loading data")
    train, test, oil, store, transaction, holiday = load_data()

    train, train_start, train_end = reindex_train(train)
    test_end = test.date.max()
    oil = fill_oil(oil, train_start, test_end)
    transaction = fill_transactions(transaction, train, store.store_nbr.nunique())
    work_days, national_holidays = process_holidays(holiday, store)

    print(f"train range: {train_start.date()} .. {train_end.date()}")
    print(f"test  range: {test.date.min().date()} .. {test_end.date()}")

    target_train_end = train_end

    print(">>> building data frame")
    data = build_data(train, test, transaction, oil, store, work_days, national_holidays,
                      train_end, s, anchor=target_train_end)

    print(">>> extracting target series")
    static_cols = ["city", "state", "type", "cluster"]
    target_dict, pipe_dict, id_dict = get_target_series(data, static_cols, target_train_end, log_transform=s.log_transform)

    print(">>> extracting covariates")
    past_cols = ["transactions"]
    future_cols = [
        "oil", "onpromotion",
        "day", "month", "year", "day_of_week", "day_of_year", "week_of_year", "date_index",
        "work_day", *SELECTED_HOLIDAYS,
    ]
    if s.school_region:
        future_cols += ["sierra_school_ramp", "sierra_late_summer"]
        print(f">>> SCHOOL_REGION=1 — added Sierra ramp/late-summer feats ({len(future_cols)} future cols)")
    if s.new_feats:
        future_cols += ["days_to_15", "days_to_eom", "is_payday", "quake_after"]
        if s.new_feats_famtotal:
            future_cols += ["fam_total"]
        print(f">>> NEW_FEATS=1 (famtotal={s.new_feats_famtotal}) — added payday/quake-decay"
              f"{'/fam_total' if s.new_feats_famtotal else ''} ({len(future_cols)} future cols)")
    if s.floor_feature:
        future_cols.append("cell_floor")
        print(f">>> FLOOR_FEATURE=1 — added cell_floor to future_cols ({len(future_cols)} total)")
    past_dict, future_dict = get_covariates(
        data, past_cols, future_cols, target_train_end,
        past_ma_cols=None, future_ma_cols=["oil", "onpromotion"],
    )

    configs = build_configs(s)
    print(f">>> SEED={s.seed} DEPTH={s.depth} N_EST={s.n_estimators} LR={s.lr} "
          f"INCLUDE_XGB={s.include_xgb} CAT_ONLY={s.cat_only}")
    print(f">>> #configs={len(configs)} OUT={out_path.name}")

    weight_dict = None
    if s.sample_weight_floor:
        print(">>> building sample_weight from cell_floor")
        weight_dict = get_weight_series(data, target_train_end, s)

    print(">>> ensemble predict on full train")
    pred_full = ensemble_predict(target_dict, pipe_dict, id_dict, past_dict, future_dict, configs, drop_before=None, weight_dict=weight_dict)

    print(">>> ensemble predict with drop_before=2015-01-01")
    pred_2015 = ensemble_predict(target_dict, pipe_dict, id_dict, past_dict, future_dict, configs, drop_before="2015-01-01", weight_dict=weight_dict)

    print(">>> merging & averaging")
    final = pred_full.merge(pred_2015, on=["date", "store_nbr", "family"], how="left")
    final["sales"] = final[["sales_x", "sales_y"]].mean(axis=1)
    final = final.drop(columns=["sales_x", "sales_y"])
    final.store_nbr = final.store_nbr.replace("store_nbr_", "", regex=True).astype(int)

    submission = test.merge(final, on=["date", "store_nbr", "family"], how="left")[["id", "sales"]]
    submission["sales"] = submission["sales"].clip(lower=0.0).fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f">>> wrote {out_path}  rows={len(submission)}  mean={submission.sales.mean():.3f}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: select a config variant and train it."""
    ap = argparse.ArgumentParser(description="darts per-family GBT family member")
    ap.add_argument("--variant", default="base",
                     choices=sorted(_cfg.darts_family.variants),
                     help="config-defined variant to train")
    args = ap.parse_args(argv)
    run(DartsSettings.from_variant(args.variant))


if __name__ == "__main__":
    main()
