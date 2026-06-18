"""Minimum-variance blend library (log1p space).

The core math behind the ensemble, imported by `build_ensemble.py`:

  * `load_log(file)`           — read a submission CSV, return log1p(sales).
  * `reconstruct_cov(...)`     — rebuild the cross-leg error covariance from
                                 LB-anchored sigmas + pairwise RMS differences,
                                 without ground truth.
  * `min_var_weights(...)`     — minimum-variance weights w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1).
  * `build_family(alpha)`      — the 6-model gradient-boosted-tree sub-blend.

Covariance reconstruction uses the identity
    Cov_ij = (σ_i² + σ_j² − D_ij²) / 2
where D_ij is the RMS difference between two legs' log-predictions. Two legs
that disagree a lot (large D) have low error correlation and combine well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

# The 6 gradient-boosted-tree submissions that make up the `family` sub-blend.
FAMILY_FILES = {
    "base": "submission_darts_lgbm.csv",
    "deeper": "submission_darts_lgbm_deeper.csv",
    "xgb": "submission_darts_xgb_lgbm.csv",
    "sub3": "submission_darts_lgbm_subsampled_3seed.csv",
    "cat_deep": "submission_darts_cat_deeper.csv",
    "weighted": "submission_darts_lgbm_w.csv",
}

# LB-anchored standalone RMSLE for each family member.
# Each value is the Kaggle leaderboard RMSLE obtained by SUBMITTING that member's
# CSV (see notebook section 4.2 "Submit & get Kaggle score"). Replace with your own scores.
FAMILY_SIGMA = {
    "base": 0.38345,
    "deeper": 0.38098,
    "xgb": 0.38231,
    "sub3": 0.38381,
    "cat_deep": 0.38182,
    "weighted": 0.38344,
}

# Family non-negative-projection mix (see build_family); alpha=0.5 → half-and-half.
FAMILY_ALPHA = 0.5

# The 4 legs of the final ensemble. "family" is the build_family() sub-blend (not a
# single CSV); the other three are single submission CSVs.
LEG_FILES = {
    "chronos2-cov": "submission_chronos2_cov_promo.csv",
    "lgbm-reg": "submission_v8_reg.csv",
    "tsmixer": "submission_tsmixer_tuned.csv",
}

# LB-anchored standalone RMSLE (sigma) for each of the 4 legs — the parameters the
# blend needs. Obtain each by SUBMITTING the leg's CSV to Kaggle (notebook §4.2):
#   * "family"       → submit submission_family.csv  (written by build_ensemble.py)
#   * "chronos2-cov" → submit submission_chronos2_cov_promo.csv
#   * "lgbm-reg"     → submit submission_v8_reg.csv
#   * "tsmixer"      → submit submission_tsmixer_tuned.csv
# Replace the reference numbers below with your own leaderboard scores.
LEG_SIGMA = {
    "family": 0.37856,
    "chronos2-cov": 0.40100,
    "lgbm-reg": 0.48297,
    "tsmixer": 0.38191,
}

# The CSV written by build_ensemble.py for the family sub-blend (so it is submittable).
FAMILY_OUT_FILE = "submission_family.csv"


def load_log(file: str) -> np.ndarray:
    """Read a submission CSV (id-sorted) and return log1p of its non-negative sales."""
    sales = (
        pd.read_csv(SUBMISSIONS / file)
        .sort_values("id")
        .reset_index(drop=True)
        .sales.clip(lower=0)
        .to_numpy()
    )
    return np.log1p(sales)


def reconstruct_cov(sigmas, log_preds):
    """Rebuild the cross-leg error covariance from sigmas + pairwise RMS differences.

    Returns (cov, rms_diff) where rms_diff[i, j] is the RMS difference between
    legs i and j in log space.
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


def min_var_weights(sigmas, log_preds):
    """Minimum-variance weights over the legs.

    Returns (weights, blended_rmsle, cov, rms_diff). Negative weights are
    allowed — they actively cancel shared error.
    """
    cov, rms_diff = reconstruct_cov(sigmas, log_preds)
    cov_inv = np.linalg.inv(cov)
    ones = np.ones(len(sigmas))
    weights = cov_inv @ ones / (ones @ cov_inv @ ones)
    blended_rmsle = float(np.sqrt(weights @ cov @ weights))
    return weights, blended_rmsle, cov, rms_diff


def build_family(alpha: float):
    """Build the 6-model gradient-boosted-tree sub-blend.

    Combines the unconstrained min-var weights with their non-negative
    projection via `alpha` (alpha=0.5 -> half-and-half) for robustness.

    Returns (family_log_pred, family_sigma).
    """
    members = list(FAMILY_FILES)
    n = len(members)
    logs = {m: load_log(FAMILY_FILES[m]) for m in members}
    sigmas = np.array([FAMILY_SIGMA[m] for m in members])

    cov, _ = reconstruct_cov(sigmas, [logs[m] for m in members])
    cov_inv = np.linalg.inv(cov)
    ones = np.ones(n)
    weights_unconstrained = cov_inv @ ones / (ones @ cov_inv @ ones)
    weights_nonneg = np.maximum(weights_unconstrained, 0)
    weights_nonneg /= weights_nonneg.sum()
    weights = (1 - alpha) * weights_nonneg + alpha * weights_unconstrained

    family_log = sum(w * logs[m] for w, m in zip(weights, members))
    family_sigma = float(np.sqrt(weights @ cov @ weights))
    return family_log, family_sigma


def _ids():
    """Canonical id column (sorted) shared by every submission file."""
    return (
        pd.read_csv(SUBMISSIONS / FAMILY_FILES["base"])
        .sort_values("id")
        .reset_index(drop=True)["id"]
    )


def family_submission(alpha: float = FAMILY_ALPHA) -> pd.DataFrame:
    """The family sub-blend as a submittable (id, sales) frame.

    Written to disk by build_ensemble.py so its standalone Kaggle RMSLE (the
    "family" leg sigma in LEG_SIGMA) can be measured by submitting it.
    """
    family_log, _ = build_family(alpha)
    return pd.DataFrame({"id": _ids(), "sales": np.expm1(family_log).clip(min=0)})


def build_fourway(leg_sigma: dict | None = None, family_alpha: float = FAMILY_ALPHA):
    """Compute the 4-way minimum-variance blend from LB-anchored sigmas.

    No ground truth is needed: the weights come from the per-leg sigmas
    (Kaggle leaderboard RMSLE) plus the pairwise prediction differences. The
    blend's formula RMSLE is an *estimate*; the official final RMSLE is whatever
    Kaggle reports when you submit the resulting 4-way CSV.

    Returns a dict with: legs, sigmas, weights, blend_rmsle_formula, rms_diff,
    log_blend, ids.
    """
    leg_sigma = leg_sigma or LEG_SIGMA
    legs = list(leg_sigma)

    family_log, _ = build_family(family_alpha)
    logs = {"family": family_log,
            **{name: load_log(LEG_FILES[name]) for name in LEG_FILES}}

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
        "ids": _ids(),
    }
