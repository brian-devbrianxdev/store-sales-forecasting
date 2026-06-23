#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Regenerate the 5 darts-family variants that still ship PRE-advanced-FE CSVs,
# then rebuild the ensemble blend. The `base` variant is already regenerated and
# submitted (LB 0.38337), so it is intentionally excluded here.
#
# Why: the advanced oil/holiday FE is ALWAYS-ON in darts_family now, so the
# `family` sub-blend is incoherent (1 new + 5 old members) until these 5 are
# re-run. This script prepares hướng (B): a coherent family ensemble.
#
# Usage:
#   ./run_darts_variants.sh                # all 5 variants, then `blend build`
#   ./run_darts_variants.sh deeper xgb     # only the named variants (no blend)
#
# Per-variant logs land in /tmp/darts_variants/<variant>.log. Runs SEQUENTIALLY
# (each leg already saturates the CPU) and fails fast (`set -e`).
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
RUN="python3 -m store_sales.cli"
LOG_DIR="/tmp/darts_variants"
mkdir -p "$LOG_DIR"

# Variant names as defined in config.yaml darts_family.variants
DEFAULT_VARIANTS=(deeper xgb subsampled cat_deep weighted)

if [ "$#" -gt 0 ]; then
  VARIANTS=("$@"); DO_BLEND=0
else
  VARIANTS=("${DEFAULT_VARIANTS[@]}"); DO_BLEND=1
fi

echo ">>> darts-family variants to run: ${VARIANTS[*]}"
echo ">>> blend build afterwards: $DO_BLEND"
START=$(date +%s)

for v in "${VARIANTS[@]}"; do
  ts=$(date +%H:%M:%S)
  echo ">>> [$ts] training variant: $v  (log: $LOG_DIR/$v.log)"
  $RUN train darts-family --variant "$v" > "$LOG_DIR/$v.log" 2>&1
  echo ">>> [$(date +%H:%M:%S)] done: $v"
done

if [ "$DO_BLEND" -eq 1 ]; then
  echo ">>> [$(date +%H:%M:%S)] building ensemble blend"
  $RUN blend build > "$LOG_DIR/blend_build.log" 2>&1
  echo ">>> blend build complete (log: $LOG_DIR/blend_build.log)"
  echo ">>> verifying blend"
  $RUN blend verify || echo "!!! blend verify reported issues — inspect output above"
fi

echo ">>> ALL DONE in $(( $(date +%s) - START ))s"
