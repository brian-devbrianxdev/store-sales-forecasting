"""Assemble the final ensemble submission and verify reproducibility.

This is the only stage that runs end-to-end without retraining: it reads the
leg CSVs, computes the minimum-variance blend, and writes the final ensemble
CSV (``ensemble.out_file``). ``--verify`` rebuilds the blend and asserts a
byte-exact match against the written file — the regression gate for the
whole pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import Config, get_config
from . import blend


def build(cfg: Config | None = None) -> pd.DataFrame:
    """Compute the blend and return it as an ``(id, sales)`` frame.

    Prints the blend weights, the formula RMSLE estimate, and the
    family↔tsmixer redundancy diagnostic.

    Args:
        cfg: Optional config; defaults to the process-wide cached config.

    Returns:
        An ``(id, sales)`` dataframe (unsorted; the caller sorts).
    """
    cfg = cfg or get_config()
    blend_result = blend.build_fourway(cfg=cfg)

    print("weights " + " ".join(
        f"{leg}={w:.4f}"
        for leg, w in zip(blend_result["legs"], blend_result["weights"])
    ))
    print(f"math_LB = {blend_result['blend_rmsle_formula']:.5f}  "
          f"(formula estimate)")
    fam = blend_result["legs"].index("family")
    tsm = blend_result["legs"].index("tsmixer")
    print(f"diff family<->tsmixer = {blend_result['rms_diff'][fam, tsm]:.3f} "
          f"(family-level redundancy watch)")

    sales = np.expm1(blend_result["log_blend"]).clip(min=0)
    return pd.DataFrame({"id": blend_result["ids"], "sales": sales})


def run_build(cfg: Config | None = None) -> Path:
    """Build the ensemble blend and the family sub-blend, writing both CSVs.

    Args:
        cfg: Optional config; defaults to the process-wide cached config.

    Returns:
        Path to the ensemble submission that was written.
    """
    cfg = cfg or get_config()
    out_file = cfg.ensemble.out_file
    paths.ensure_dirs()

    frame = build(cfg).sort_values("id").reset_index(drop=True)
    frame.to_csv(paths.SUBMISSIONS / out_file, index=False)
    print(f"Wrote {out_file}  (rows={len(frame)}, max sales={frame.sales.max():.0f})")

    # Also emit the family sub-blend so its standalone leg sigma is submittable.
    fam = blend.family_submission(cfg=cfg).sort_values("id").reset_index(drop=True)
    fam.to_csv(paths.SUBMISSIONS / cfg.ensemble.family_out_file, index=False)
    print(f"Wrote {cfg.ensemble.family_out_file}  (rows={len(fam)})")
    return paths.SUBMISSIONS / out_file


def verify(cfg: Config | None = None) -> bool:
    """Rebuild the blend and assert a byte-exact match vs the on-disk CSV.

    Args:
        cfg: Optional config; defaults to the process-wide cached config.

    Returns:
        ``True`` iff the rebuilt blend matches the on-disk submission to
        within ``1e-6`` absolute on every row.
    """
    cfg = cfg or get_config()
    frame = build(cfg).sort_values("id").reset_index(drop=True)
    ref = (pd.read_csv(paths.SUBMISSIONS / cfg.ensemble.out_file)
           .sort_values("id").reset_index(drop=True))
    max_delta = float(np.abs(frame.sales.values - ref.sales.values).max())
    verdict = "EXACT MATCH" if max_delta < 1e-6 else "DIFFERS"
    print(f"VERIFY max|delta| = {max_delta:.3e}  ->  {verdict}")
    return max_delta < 1e-6


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build the blend, or verify reproduction with ``--verify``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true",
        help="rebuild and assert byte-exact match vs the on-disk submission; "
             "do not overwrite",
    )
    args = parser.parse_args(argv)

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)
    run_build()


if __name__ == "__main__":
    main()
