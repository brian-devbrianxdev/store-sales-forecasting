"""Typed configuration loaded from ``config.yaml``.

A single :func:`load_config` call parses the YAML into frozen dataclasses so the
rest of the package gets attribute access (``cfg.common.horizon``) with IDE
completion and type checking, instead of scattering string-keyed dict lookups.

Date-like strings in the ``common`` section are converted to
:class:`pandas.Timestamp` at load time so callers never re-parse them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import paths


@dataclass(frozen=True)
class Common:
    horizon: int
    context_len: int
    train_from: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    earthquake_date: pd.Timestamp


@dataclass(frozen=True)
class LgbmV8:
    seeds_pool: list[int]
    default_seeds: int
    fixed_iter_mult: float
    lags_daily: list[int]
    lags_long: list[int]
    roll_windows: list[int]
    ewm_halflives: list[int]
    hol_leadlag: list[int]
    dow_k_values: list[int]
    per_h_roll_bases: list[int]
    params: dict[str, Any]
    tweedie: dict[str, Any]

    @property
    def all_lags(self) -> list[int]:
        """Daily + long lags concatenated (the original ``ALL_LAGS``)."""
        return list(self.lags_daily) + list(self.lags_long)


@dataclass(frozen=True)
class Catboost:
    seeds_pool: list[int]
    default_seeds: int
    default_suffix: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DartsFamily:
    forecast_horizon: int
    zero_fc_window: int
    selected_holidays: list[str]
    sierra_states: list[str]
    base: dict[str, Any]
    variants: dict[str, dict[str, Any]]

    def variant(self, name: str) -> dict[str, Any]:
        """Return the fully-resolved settings for one named variant.

        The ``base`` block holds every default; the named variant supplies only
        its overrides. The merged dict reproduces the env-var combination the
        notebook used for that submission.

        Args:
            name: One of the keys in ``variants`` (e.g. ``"deeper"``).

        Returns:
            ``base`` merged with the variant's overrides.

        Raises:
            KeyError: If ``name`` is not a known variant.
        """
        if name not in self.variants:
            raise KeyError(
                f"unknown darts_family variant {name!r}; "
                f"choices: {sorted(self.variants)}"
            )
        return {**self.base, **self.variants[name]}


@dataclass(frozen=True)
class Neural:
    default_model: str
    default_epochs: int
    seed: int
    input_chunk_length: int
    batch_size: int
    use_foy: bool
    out_name_template: str
    tuned_out_name: str
    tsmixer: dict[str, Any]
    tide: dict[str, Any]
    nhits: dict[str, Any]


@dataclass(frozen=True)
class Chronos:
    model: str
    plain_batch_size: int
    cov_batch_size: int


@dataclass(frozen=True)
class Ensemble:
    family_alpha: float
    family_out_file: str
    out_file: str
    family_files: dict[str, str]
    family_sigma: dict[str, float]
    leg_files: dict[str, str]
    leg_sigma: dict[str, float]
    cov_oilhol_file: str
    cov_oilhol_sigma: float
    swap_out_file: str
    hedge_out_file: str


@dataclass(frozen=True)
class Config:
    common: Common
    lgbm_v8: LgbmV8
    catboost: Catboost
    darts_family: DartsFamily
    neural: Neural
    chronos: Chronos
    ensemble: Ensemble


def _to_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def load_config(path: Path | str | None = None) -> Config:
    """Parse ``config.yaml`` into a :class:`Config`.

    Args:
        path: Optional explicit path to the YAML file. Defaults to
            :data:`store_sales.paths.CONFIG_FILE`.

    Returns:
        The parsed, typed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = Path(path) if path is not None else paths.CONFIG_FILE
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)

    common = Common(
        horizon=int(raw["common"]["horizon"]),
        context_len=int(raw["common"]["context_len"]),
        train_from=_to_ts(raw["common"]["train_from"]),
        val_start=_to_ts(raw["common"]["val_start"]),
        test_start=_to_ts(raw["common"]["test_start"]),
        test_end=_to_ts(raw["common"]["test_end"]),
        earthquake_date=_to_ts(raw["common"]["earthquake_date"]),
    )
    lgbm_v8 = LgbmV8(**raw["lgbm_v8"])
    catboost = Catboost(**raw["catboost"])
    darts_family = DartsFamily(**raw["darts_family"])
    neural = Neural(**raw["neural"])
    chronos = Chronos(**raw["chronos"])
    ensemble = Ensemble(**raw["ensemble"])

    return Config(
        common=common,
        lgbm_v8=lgbm_v8,
        catboost=catboost,
        darts_family=darts_family,
        neural=neural,
        chronos=chronos,
        ensemble=ensemble,
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a process-wide cached :class:`Config` (loaded once)."""
    return load_config()
