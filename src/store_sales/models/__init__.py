"""The four training legs of the ensemble.

Each module is runnable through the CLI (:mod:`store_sales.cli`):

* :mod:`store_sales.models.lgbm_regularized` — regularized per-family LightGBM v8.
* :mod:`store_sales.models.catboost_family`  — CatBoost family member.
* :mod:`store_sales.models.darts_family`     — darts per-family GBT family (6 variants).
* :mod:`store_sales.models.neural_ts`        — global neural forecaster (TSMixer/TiDE/NHiTS).
* :mod:`store_sales.models.chronos2`         — Chronos-2 zero-shot.
* :mod:`store_sales.models.chronos2_cov`     — Chronos-2 with covariates.

The Chronos and darts/neural legs require heavy optional dependencies
(``chronos-forecasting``, ``darts``, ``torch``); importing this package does not
import them — they are imported lazily inside each leg's ``main``.
"""
from __future__ import annotations
