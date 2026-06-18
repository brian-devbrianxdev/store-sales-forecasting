"""Evaluation metric — the single definition of RMSLE.

The competition is scored on Root Mean Squared Logarithmic Error. The former
scripts each carried their own copy of this function; the body here is identical
to the original ``lgbm_regularized.rmsle`` so every leg now shares one
implementation.
"""
from __future__ import annotations

import numpy as np


def rmsle(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error on raw (non-log) sales.

    Both inputs are clipped to be non-negative before the log transform, so
    negative predictions do not raise. Computed as
    ``sqrt(mean((log1p(pred) - log1p(true))**2))``.

    Args:
        y_true_raw: Ground-truth sales, shape ``(n,)``.
        y_pred_raw: Predicted sales, shape ``(n,)``.

    Returns:
        The scalar RMSLE.
    """
    y_true_raw = np.clip(y_true_raw, 0, None)
    y_pred_raw = np.clip(y_pred_raw, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true_raw) - np.log1p(y_pred_raw)) ** 2)))
