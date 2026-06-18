from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import blend

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"
OUT_FILE = "submission_fam_cov_v8_tsmTuned_4way.csv"


def build() -> pd.DataFrame:
    r = blend.build_fourway()

    print("weights " + " ".join(f"{leg}={w:.4f}" for leg, w in zip(r["legs"], r["weights"])))
    print(f"math_LB = {r['blend_rmsle_formula']:.5f}  [reference actual LB 0.37418]")
    fam, tsm = r["legs"].index("family"), r["legs"].index("tsmixer")
    print(f"diff family<->tsmixer = {r['rms_diff'][fam, tsm]:.3f} (family-level redundancy watch)")

    sales = np.expm1(r["log_blend"]).clip(min=0)
    return pd.DataFrame({"id": r["ids"], "sales": sales})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify", action="store_true",
        help="rebuild and assert byte-exact match vs the committed submission; do not overwrite",
    )
    args = parser.parse_args()

    df = build().sort_values("id").reset_index(drop=True)
    if args.verify:
        ref = pd.read_csv(SUBMISSIONS / OUT_FILE).sort_values("id").reset_index(drop=True)
        max_delta = float(np.abs(df.sales.values - ref.sales.values).max())
        verdict = "EXACT MATCH" if max_delta < 1e-6 else "DIFFERS"
        print(f"VERIFY max|delta| = {max_delta:.3e}  ->  {verdict}")
        sys.exit(0 if max_delta < 1e-6 else 1)

    df.to_csv(SUBMISSIONS / OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}  (rows={len(df)}, max sales={df.sales.max():.0f})")

    # Also emit the family sub-blend so its standalone leg sigma is submittable.
    fam = blend.family_submission().sort_values("id").reset_index(drop=True)
    fam.to_csv(SUBMISSIONS / blend.FAMILY_OUT_FILE, index=False)
    print(f"Wrote {blend.FAMILY_OUT_FILE}  (rows={len(fam)})")


if __name__ == "__main__":
    main()
