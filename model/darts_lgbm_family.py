"""Per-family LightGBM family members (darts-based).

Recipe: per-family LightGBM ensemble of 4 lag configs (lags=56,7,365,730),
trained on full data AND 2015+ subset, averaged. Each member is selected via
env vars (OUT_NAME, OBJECTIVE, DEPTH, CAT_ONLY, INCLUDE_XGB, ...); together
they form the `family` sub-blend in build_ensemble.py.

Usage:
    .venv/bin/python3 src/darts_lgbm_family.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

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

REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "data"

# Config via env vars (with defaults matching the public-notebook recipe).
SEED = int(os.environ.get("SEED", "0"))
DEPTH = int(os.environ.get("DEPTH", "6"))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "100"))
LR = float(os.environ.get("LR", "0.065"))
INCLUDE_XGB = os.environ.get("INCLUDE_XGB", "0") == "1"
# CAT_ONLY=1 → swap LightGBMModel for CatBoostModel in the 4 lag-configs ensemble.
CAT_ONLY = os.environ.get("CAT_ONLY", "0") == "1"
FEATURE_FRACTION = float(os.environ.get("FEATURE_FRACTION", "1.0"))
BAGGING_FRACTION = float(os.environ.get("BAGGING_FRACTION", "1.0"))
BAGGING_FREQ = int(os.environ.get("BAGGING_FREQ", "0"))
# OBJECTIVE="tweedie" → LightGBM tweedie loss on RAW (non-log) target for zero-heavy
# retail (Kaggle 1st-place recipe). Pair with LOG_TRANSFORM=0 to drop the log1p mapper
# so tweedie's log-link models raw sales directly (orthogonal-leg probe vs log1p-MSE trees).
OBJECTIVE = os.environ.get("OBJECTIVE", "")
TWEEDIE_VARIANCE_POWER = float(os.environ.get("TWEEDIE_VARIANCE_POWER", "1.2"))
LOG_TRANSFORM = os.environ.get("LOG_TRANSFORM", "1") == "1"
OUT_NAME = os.environ.get("OUT_NAME", "submission_darts_lgbm.csv")
OUT = REPO / "submissions" / OUT_NAME
# FLOOR_FEATURE=1 → merge data/floor_per_row.parquet and add cell_floor to future covs.
FLOOR_FEATURE = os.environ.get("FLOOR_FEATURE", "0") == "1"
# SAMPLE_WEIGHT_FLOOR=1 → use 1/sqrt(cell_floor+0.1) as sample weights during fit.
SAMPLE_WEIGHT_FLOOR = os.environ.get("SAMPLE_WEIGHT_FLOOR", "0") == "1"
# NEW_FEATS=1 → add genuinely-new information vs the existing feature set (earthquake is
# already in via nat_terremoto; payday ⊂ day-of-month): national family demand (cross-series,
# ffilled→future cov), explicit payday-distance shape, and extended 2016 quake-decay window.
NEW_FEATS = os.environ.get("NEW_FEATS", "0") == "1"
# NEW_FEATS_FAMTOTAL=0 → keep only the deterministic new feats (payday/quake), drop the
# cross-series fam_total future-cov (ablation: it ffills flat over the horizon -> train/serve shift).
NEW_FEATS_FAMTOTAL = os.environ.get("NEW_FEATS_FAMTOTAL", "1") == "1"
# WEIGHT_SCHEME: sqrt | linear | sq — only effective if SAMPLE_WEIGHT_FLOOR=1.
WEIGHT_SCHEME = os.environ.get("WEIGHT_SCHEME", "sqrt")
# SCHOOL_REGION=1 → add Sierra-region × late-summer future covariates. National calendar
# feats (month/day_of_year) can't express region-conditional seasonality, and `state` is too
# high-cardinality for the per-family tree to split on August×state (1/12 × 16 states, sparse).
# Collapsing to a low-card Sierra gate × a Sep-1 ramp lets each family learn its own response:
# SCHOOL & OFFICE ramps UP in the Sierra (school year starts ~Sep), GROCERY II dampens —
# both region-conditional biases confirmed reproducible in 2015/2016 August train residuals.
SCHOOL_REGION = os.environ.get("SCHOOL_REGION", "0") == "1"


# -------------------- Data loading & preprocessing --------------------

def load_data():
    train = pd.read_csv(PATH / "train.csv", parse_dates=["date"])
    test = pd.read_csv(PATH / "test.csv", parse_dates=["date"])
    oil = pd.read_csv(PATH / "oil.csv", parse_dates=["date"]).rename(
        columns={"dcoilwtico": "oil"}
    )
    store = pd.read_csv(PATH / "stores.csv")
    transaction = pd.read_csv(PATH / "transactions.csv", parse_dates=["date"])
    holiday = pd.read_csv(PATH / "holidays_events.csv", parse_dates=["date"])
    return train, test, oil, store, transaction, holiday


def reindex_train(train):
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


def fill_oil(oil, train_start, test_end):
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


def fill_transactions(transaction, train, num_store):
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


def process_holidays(holiday, store):
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

SELECTED_HOLIDAYS = [
    "nat_terremoto", "nat_navidad", "nat_dia la madre", "nat_dia trabajo",
    "nat_primer dia ano", "nat_futbol", "nat_dia difuntos",
]

# Andes/Amazon provinces on the Sierra school calendar (year starts ~Sep → August
# back-to-school surge). Coast provinces start ~May, so no August surge. Spellings match
# stores.csv exactly (verified: every store state falls in Sierra ∪ Coast). SCHOOL_REGION only.
SIERRA_STATES = {
    "Pichincha", "Cotopaxi", "Tungurahua", "Chimborazo", "Imbabura",
    "Bolivar", "Carchi", "Cañar", "Azuay", "Loja", "Pastaza",
}


def build_data(train, test, transaction, oil, store, work_days, national_holidays, train_end, anchor=None):
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

    if FLOOR_FEATURE or SAMPLE_WEIGHT_FLOOR:
        floor = pd.read_parquet(PATH / "floor_per_row.parquet")[
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

    if SCHOOL_REGION:
        # `state` is still raw here (stringified below). Sierra gate is constant-in-time per
        # store → it would vanish under the per-series Scaler; so encode it only via TIME-VARYING
        # interactions that survive scaling. Tree learns per-family sign/scale.
        sierra = data.state.isin(SIERRA_STATES).to_numpy(dtype=np.float32)
        doy = data.date.dt.dayofyear.to_numpy()
        # triangular bump peaking at Sep-1 (doy≈244), ±45-day support: rises through August
        # (covers val Aug 1-15 AND the late-Aug LB window), 0 outside Jul18–Oct16.
        ramp = np.clip(1.0 - np.abs(doy - 244) / 45.0, 0.0, None).astype(np.float32)
        data["sierra_school_ramp"] = sierra * ramp
        # broader Aug/Sep level handle (not Sep-1-peaked) for the GROCERY II damping.
        in_late_summer = data.date.dt.month.isin([8, 9]).to_numpy(dtype=np.float32)
        data["sierra_late_summer"] = sierra * in_late_summer

    if NEW_FEATS:
        dom = data.date.dt.day
        eom = data.date.dt.daysinmonth
        # explicit payday shape (Ecuador public sector paid 15th + month-end); day-of-month
        # is already a feature, this hands the bump distance directly.
        data["days_to_15"] = (dom - 15).abs().astype(np.float32)
        data["days_to_eom"] = (eom - dom).astype(np.float32)
        data["is_payday"] = ((dom == 15) | (dom == eom)).astype(np.float32)
        # extended 2016 earthquake aftermath: nat_terremoto flags only the official day(s);
        # relief-buying ran ~6 weeks. Decaying exogenous shock, known-future (0 across 2017).
        dq = (data.date - pd.Timestamp("2016-04-16")).dt.days
        data["quake_after"] = np.where((dq >= 0) & (dq <= 45), np.exp(-dq / 21.0), 0.0).astype(np.float32)
        # cross-series NEW info: national family demand (sum over stores), log1p, as a future
        # covariate. LEAK-PROOF: sales strictly AFTER the forecast anchor are masked before the
        # sum, then ffilled -> the horizon carries the last-known (anchor) family level only.
        # (Without the mask, build_data sees the val/test window's real sales and leaks the
        # target, since darts auto-regression reaches future past/future-cov values.)
        if NEW_FEATS_FAMTOTAL:
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

def get_pipeline(static_covs_transform=False, log_transform=False):
    lst = [MissingValuesFiller(n_jobs=-1)]
    if static_covs_transform:
        lst.append(StaticCovariatesTransformer(transformer_cat=OneHotEncoder(), n_jobs=-1))
    if log_transform:
        lst.append(InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, n_jobs=-1))
    lst.append(Scaler())
    return Pipeline(lst)


def get_target_series(data, static_cols, train_end, log_transform=True):
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


def get_weight_series(data, train_end):
    """Per-(store, family) sample_weight TimeSeries from cell_floor.

    Scheme determined by WEIGHT_SCHEME env var:
      - sqrt   (default): 1 / sqrt(cell_floor + 0.1)
      - linear:           1 / (cell_floor + 0.1)
      - sq:               1 / (cell_floor**2 + 0.1)
    """
    weight_dict = {}
    for fam in tqdm(data.family.unique(), desc=f"Weights({WEIGHT_SCHEME})"):
        df = data[(data.family.eq(fam)) & (data.date.le(train_end.strftime("%Y-%m-%d")))].copy()
        cf = df["cell_floor"].astype(float)
        if WEIGHT_SCHEME == "linear":
            df["weight"] = (1.0 / (cf + 0.1)).clip(0.1, 10.0)
        elif WEIGHT_SCHEME == "sq":
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

FORECAST_HORIZON = 16
ZERO_FC_WINDOW = 21


def clip_nonneg(arr):
    return np.clip(arr, a_min=0.0, a_max=None)


def get_models(configs):
    out = []
    for cfg in configs:
        cfg = dict(cfg)
        cls = cfg.pop("_cls", LightGBMModel)
        out.append(cls(**cfg))
    return out


def generate_forecasts(models, train, pipe, past_covs, future_covs, drop_before=None, weights=None):
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


def ensemble_predict(target_dict, pipe_dict, id_dict, past_dict, future_dict, configs, drop_before=None, weight_dict=None):
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


# -------------------- Main --------------------

def main():
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
    data = build_data(train, test, transaction, oil, store, work_days, national_holidays, train_end,
                      anchor=target_train_end)

    print(">>> extracting target series")
    static_cols = ["city", "state", "type", "cluster"]
    target_dict, pipe_dict, id_dict = get_target_series(data, static_cols, target_train_end, log_transform=LOG_TRANSFORM)

    print(">>> extracting covariates")
    past_cols = ["transactions"]
    future_cols = [
        "oil", "onpromotion",
        "day", "month", "year", "day_of_week", "day_of_year", "week_of_year", "date_index",
        "work_day", *SELECTED_HOLIDAYS,
    ]
    if SCHOOL_REGION:
        future_cols += ["sierra_school_ramp", "sierra_late_summer"]
        print(f">>> SCHOOL_REGION=1 — added Sierra ramp/late-summer feats ({len(future_cols)} future cols)")
    if NEW_FEATS:
        future_cols += ["days_to_15", "days_to_eom", "is_payday", "quake_after"]
        if NEW_FEATS_FAMTOTAL:
            future_cols += ["fam_total"]
        print(f">>> NEW_FEATS=1 (famtotal={NEW_FEATS_FAMTOTAL}) — added payday/quake-decay"
              f"{'/fam_total' if NEW_FEATS_FAMTOTAL else ''} ({len(future_cols)} future cols)")
    if FLOOR_FEATURE:
        future_cols.append("cell_floor")
        print(f">>> FLOOR_FEATURE=1 — added cell_floor to future_cols ({len(future_cols)} total)")
    past_dict, future_dict = get_covariates(
        data, past_cols, future_cols, target_train_end,
        past_ma_cols=None, future_ma_cols=["oil", "onpromotion"],
    )

    lgbm_extra = {}
    if FEATURE_FRACTION < 1.0:
        lgbm_extra["feature_fraction"] = FEATURE_FRACTION
    if BAGGING_FRACTION < 1.0 and BAGGING_FREQ > 0:
        lgbm_extra["bagging_fraction"] = BAGGING_FRACTION
        lgbm_extra["bagging_freq"] = BAGGING_FREQ
    base_config = {
        "random_state": SEED,
        "lags": 56,
        "lags_past_covariates": list(range(-17, -24, -1)),
        "lags_future_covariates": (14, 1),
        "output_chunk_length": 1,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LR,
        "max_depth": DEPTH,
    }
    if OBJECTIVE and not CAT_ONLY:
        base_config["objective"] = OBJECTIVE
        if OBJECTIVE == "tweedie":
            base_config["tweedie_variance_power"] = TWEEDIE_VARIANCE_POWER
    if CAT_ONLY:
        # CatBoost: drop LightGBM-specific feature/bagging params; supply CatBoost-compatible ones.
        base_config["_cls"] = CatBoostModel
        # CatBoost's verbose=False to suppress per-iter logs
        base_config["verbose"] = False
    else:
        base_config.update(lgbm_extra)
    configs = [
        base_config,
        {**base_config, "lags": 7},
        {**base_config, "lags": 365},
        {**base_config, "lags": 730},
    ]
    if INCLUDE_XGB:
        xgb_base = {
            "random_state": SEED,
            "lags": 56,
            "lags_past_covariates": list(range(-17, -24, -1)),
            "lags_future_covariates": (14, 1),
            "output_chunk_length": 1,
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LR,
            "max_depth": DEPTH,
            "tree_method": "hist",
            "_cls": XGBModel,
        }
        configs += [
            xgb_base,
            {**xgb_base, "lags": 7},
            {**xgb_base, "lags": 365},
            {**xgb_base, "lags": 730},
        ]
    print(f">>> SEED={SEED} DEPTH={DEPTH} N_EST={N_ESTIMATORS} LR={LR} INCLUDE_XGB={INCLUDE_XGB} CAT_ONLY={CAT_ONLY}")
    print(f">>> #configs={len(configs)} OUT={OUT.name}")

    weight_dict = None
    if SAMPLE_WEIGHT_FLOOR:
        print(">>> building sample_weight from cell_floor")
        weight_dict = get_weight_series(data, target_train_end)

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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT, index=False)
    print(f">>> wrote {OUT}  rows={len(submission)}  mean={submission.sales.mean():.3f}")


if __name__ == "__main__":
    main()
