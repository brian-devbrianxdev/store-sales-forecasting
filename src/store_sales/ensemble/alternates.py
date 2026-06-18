"""Alternate / hedge blends beyond the main 4-way champion.

These reproduce the two extra submissions the upstream repo shipped as one-off
experiments alongside the champion:

* :func:`oilhol_swap` → ``submission_champ_oilholcov_swap.csv``
  The champion blend with its Chronos-2 covariate leg swapped from the *promo*
  variant to the *oil+holiday* variant, keeping the champion's min-variance
  weights. Reproduces the committed file to ~0.04 max sales (≈2e-6 relative);
  the tiny residual is historical ad-hoc numerics.

* :func:`positive_hedge` → ``submission_positive_4way_fam_cov_tsm.csv``
  The non-negative ("positive projection") min-variance blend over
  ``[family, cov-oilhol, v8, tsmixer]`` — the documented hedge recipe. NOTE: the
  committed upstream file was hand-calibrated on 2026-06-08 with a sigma set that
  was never recorded in the repo, so this is a faithful implementation of the
  *documented method* but does not reproduce that one-off file byte-for-byte.
  The committed original is preserved; regenerate only if you want the recipe.

Both require the oil+holiday Chronos-2 leg CSV
(``submission_chronos2_cov_promo_oil_hol.csv``) to be present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..config import Config, get_config
from ..io.submissions import load_log
from . import blend


def _oilhol_legs(cfg: Config) -> list[np.ndarray]:
    """Return the 4 leg log-predictions in champion order, with cov = oil+holiday."""
    family_log, _ = blend.build_family(cfg.ensemble.family_alpha, cfg)
    cov_oilhol = load_log(cfg.ensemble.cov_oilhol_file)
    v8 = load_log(cfg.ensemble.leg_files["lgbm-reg"])
    tsm = load_log(cfg.ensemble.leg_files["tsmixer"])
    return [family_log, cov_oilhol, v8, tsm]


def oilhol_swap(cfg: Config | None = None) -> pd.DataFrame:
    """Champion blend with the cov leg swapped to the oil+holiday variant.

    Uses the champion's minimum-variance weights (from :func:`blend.build_fourway`,
    leg order ``family / chronos2-cov / lgbm-reg / tsmixer``) but substitutes the
    oil+holiday Chronos-2 predictions for the promo ones.

    Args:
        cfg: Optional config; defaults to the cached config.

    Returns:
        An ``(id, sales)`` dataframe.
    """
    cfg = cfg or get_config()
    weights = blend.build_fourway(cfg=cfg)["weights"]
    legs = _oilhol_legs(cfg)
    log_blend = sum(w * leg for w, leg in zip(weights, legs))
    sales = np.expm1(log_blend).clip(min=0)
    return pd.DataFrame({"id": blend._ids(cfg), "sales": sales})


def positive_hedge(cfg: Config | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Non-negative min-variance hedge over ``[family, cov-oilhol, v8, tsmixer]``.

    Computes the unconstrained min-var weights, clips negatives to zero, and
    renormalises (the "positive projection"). See the module docstring for the
    byte-exactness caveat vs the committed upstream file.

    Args:
        cfg: Optional config; defaults to the cached config.

    Returns:
        ``(frame, weights)`` — the ``(id, sales)`` dataframe and the projected
        non-negative weights.
    """
    cfg = cfg or get_config()
    legs = _oilhol_legs(cfg)
    sigmas = [
        cfg.ensemble.leg_sigma["family"],
        cfg.ensemble.cov_oilhol_sigma,
        cfg.ensemble.leg_sigma["lgbm-reg"],
        cfg.ensemble.leg_sigma["tsmixer"],
    ]
    weights, _, _, _ = blend.min_var_weights(sigmas, legs)
    weights_pos = np.maximum(weights, 0)
    weights_pos = weights_pos / weights_pos.sum()
    log_blend = sum(w * leg for w, leg in zip(weights_pos, legs))
    sales = np.expm1(log_blend).clip(min=0)
    return pd.DataFrame({"id": blend._ids(cfg), "sales": sales}), weights_pos


def run_oilhol_swap(cfg: Config | None = None) -> "paths.Path":
    """Build and write the oil+holiday-swap submission."""
    cfg = cfg or get_config()
    paths.ensure_dirs()
    frame = oilhol_swap(cfg).sort_values("id").reset_index(drop=True)
    out = paths.SUBMISSIONS / cfg.ensemble.swap_out_file
    frame.to_csv(out, index=False)
    print(f"Wrote {cfg.ensemble.swap_out_file}  (rows={len(frame)}, max sales={frame.sales.max():.0f})")
    return out


def run_positive_hedge(cfg: Config | None = None) -> "paths.Path":
    """Build and write the non-negative hedge submission."""
    cfg = cfg or get_config()
    paths.ensure_dirs()
    frame, weights = positive_hedge(cfg)
    frame = frame.sort_values("id").reset_index(drop=True)
    out = paths.SUBMISSIONS / cfg.ensemble.hedge_out_file
    frame.to_csv(out, index=False)
    print(f"weights fam/cov-oilhol/v8/tsm = {np.round(weights, 4)}")
    print(f"Wrote {cfg.ensemble.hedge_out_file}  (rows={len(frame)}, max sales={frame.sales.max():.0f})")
    return out
