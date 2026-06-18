"""store_sales — modular pipeline for the Kaggle *Store Sales — Time Series
Forecasting* competition.

The package replaces the former flat ``model/`` scripts. It is organised as:

* :mod:`store_sales.config`     — typed configuration loaded from ``config.yaml``.
* :mod:`store_sales.paths`      — single source of truth for repo directories.
* :mod:`store_sales.metrics`    — the RMSLE metric (one definition).
* :mod:`store_sales.io`         — raw-data loading and submission read/write.
* :mod:`store_sales.features`   — calendar/holiday/oil feature engineering.
* :mod:`store_sales.models`     — the four training legs.
* :mod:`store_sales.ensemble`   — minimum-variance blend and final build.
* :mod:`store_sales.cli`        — command-line orchestrator.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
