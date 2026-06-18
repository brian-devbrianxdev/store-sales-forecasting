"""Minimum-variance blend library (log1p space).

The core math behind the ensemble:

  * :func:`reconstruct_cov` — rebuild the cross-leg error covariance from
    LB-anchored sigmas + pairwise RMS differences, without ground truth.
  * :func:`min_var_weights` — minimum-variance weights ``w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)``.
  * :func:`build_family`    — the 6-model gradient-boosted-tree sub-blend.
  * :func:`build_fourway`   — the final 4-leg blend.

Covariance reconstruction uses the identity
    ``Cov_ij = (σ_i² + σ_j² − D_ij²) / 2``
where ``D_ij`` is the RMS difference between two legs' log-predictions. Two legs
that disagree a lot (large ``D``) have low error correlation and combine well.

File maps and sigma values come from ``config.yaml`` (the ``ensemble`` section);
the numerical recipe is byte-for-byte identical to the former ``model/blend.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..io.submissions import canonical_ids, load_log


def reconstruct_cov(
    sigmas: list[float] | np.ndarray, log_preds: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the cross-leg error covariance from sigmas + pairwise RMS diffs.

    Args:
        sigmas: Per-leg standalone RMSLE (leaderboard sigma), length ``n``.
        log_preds: Each leg's ``log1p`` predictions; ``n`` arrays of equal length.

    Returns:
        ``(cov, rms_diff)`` where ``cov`` is the ``(n, n)`` covariance matrix and
        ``rms_diff[i, j]`` is the RMS difference between legs ``i`` and ``j`` in
        log space.
    """
    n = len(sigmas)
    rms_diff = np.array([
        [0.0 if i == j else np.sqrt(np.mean((log_preds[i] - log_preds[j]) ** 2))
         for j in range(n)]
        for i in range(n)
    ])
    cov = np.array([
        [sigmas[i] ** 2 if i == j
         else (sigmas[i] ** 2 + sigmas[j] ** 2 - rms_diff[i, j] ** 2) / 2
         for j in range(n)]
        for i in range(n)
    ])
    return cov, rms_diff


def min_var_weights(
    sigmas: list[float] | np.ndarray, log_preds: list[np.ndarray]
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Minimum-variance weights over the legs.

    Args:
        sigmas: Per-leg standalone RMSLE, length ``n``.
        log_preds: Each leg's ``log1p`` predictions; ``n`` arrays of equal length.

    Returns:
        ``(weights, blended_rmsle, cov, rms_diff)``. Negative weights are
        allowed — they actively cancel shared error.
    """
    cov, rms_diff = reconstruct_cov(sigmas, log_preds)
    cov_inv = np.linalg.inv(cov)
    ones = np.ones(len(sigmas))
    weights = cov_inv @ ones / (ones @ cov_inv @ ones)
    blended_rmsle = float(np.sqrt(weights @ cov @ weights))
    return weights, blended_rmsle, cov, rms_diff


def build_family(
    alpha: float, cfg: Config | None = None
) -> tuple[np.ndarray, float]:
    """Build the 6-model gradient-boosted-tree sub-blend.

    Combines the unconstrained min-var weights with their non-negative
    projection via ``alpha`` (``alpha=0.5`` → half-and-half) for robustness.

    Args:
        alpha: Mixing weight between the non-negative projection and the
            unconstrained min-var weights.
        cfg: Optional config; defaults to the process-wide cached config.

    Returns:
        ``(family_log_pred, family_sigma)``.
    """
    cfg = cfg or get_config()
    family_files = cfg.ensemble.family_files
    family_sigma = cfg.ensemble.family_sigma

    members = list(family_files)
    n = len(members)
    logs = {m: load_log(family_files[m]) for m in members}
    sigmas = np.array([family_sigma[m] for m in members])

    cov, _ = reconstruct_cov(sigmas, [logs[m] for m in members])
    cov_inv = np.linalg.inv(cov)
    ones = np.ones(n)
    weights_unconstrained = cov_inv @ ones / (ones @ cov_inv @ ones)
    weights_nonneg = np.maximum(weights_unconstrained, 0)
    weights_nonneg /= weights_nonneg.sum()
    weights = (1 - alpha) * weights_nonneg + alpha * weights_unconstrained

    family_log = sum(w * logs[m] for w, m in zip(weights, members))
    family_sigma_blend = float(np.sqrt(weights @ cov @ weights))
    return family_log, family_sigma_blend


def _ids(cfg: Config) -> pd.Series:
    """Canonical id column (sorted) shared by every submission file."""
    return canonical_ids(cfg.ensemble.family_files["base"])


def family_submission(
    alpha: float | None = None, cfg: Config | None = None
) -> pd.DataFrame:
    """The family sub-blend as a submittable ``(id, sales)`` frame.

    Written to disk by :mod:`store_sales.ensemble.build` so its standalone
    Kaggle RMSLE (the ``family`` leg sigma) can be measured by submitting it.

    Args:
        alpha: Family mixing weight; defaults to ``ensemble.family_alpha``.
        cfg: Optional config; defaults to the process-wide cached config.

    Returns:
        An ``(id, sales)`` dataframe.
    """
    cfg = cfg or get_config()
    if alpha is None:
        alpha = cfg.ensemble.family_alpha
    family_log, _ = build_family(alpha, cfg)
    return pd.DataFrame({"id": _ids(cfg), "sales": np.expm1(family_log).clip(min=0)})


def build_fourway(
    leg_sigma: dict[str, float] | None = None,
    family_alpha: float | None = None,
    cfg: Config | None = None,
) -> dict:
    """Compute the 4-way minimum-variance blend from LB-anchored sigmas.

    No ground truth is needed: the weights come from the per-leg sigmas (Kaggle
    leaderboard RMSLE) plus the pairwise prediction differences. The blend's
    formula RMSLE is an *estimate*; the official final RMSLE is whatever Kaggle
    reports when you submit the resulting 4-way CSV.

    Args:
        leg_sigma: Optional override of the per-leg sigma map; defaults to
            ``ensemble.leg_sigma``. Its key order defines the leg order.
        family_alpha: Family mixing weight; defaults to ``ensemble.family_alpha``.
        cfg: Optional config; defaults to the process-wide cached config.

    Returns:
        A dict with: ``legs``, ``sigmas``, ``weights``, ``blend_rmsle_formula``,
        ``rms_diff``, ``log_blend``, ``ids``.
    """
    cfg = cfg or get_config()
    leg_sigma = leg_sigma or cfg.ensemble.leg_sigma
    if family_alpha is None:
        family_alpha = cfg.ensemble.family_alpha
    leg_files = cfg.ensemble.leg_files
    legs = list(leg_sigma)

    family_log, _ = build_family(family_alpha, cfg)
    logs = {"family": family_log,
            **{name: load_log(leg_files[name]) for name in leg_files}}

    log_list = [logs[leg] for leg in legs]
    sigmas = [leg_sigma[leg] for leg in legs]
    weights, blend_rmsle, _, rms_diff = min_var_weights(sigmas, log_list)
    log_blend = sum(w * lg for w, lg in zip(weights, log_list))

    return {
        "legs": legs,
        "sigmas": np.array(sigmas),
        "weights": weights,
        "blend_rmsle_formula": blend_rmsle,
        "rms_diff": rms_diff,
        "log_blend": log_blend,
        "ids": _ids(cfg),
    }
