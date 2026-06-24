"""Per-cell sales "floor" artifact used by the darts ``weighted`` variant.

The ``weighted`` variant builds per-row sample weights ``1/sqrt(cell_floor +
0.1)`` from ``data/floor_per_row.parquet`` (column ``cell_floor`` keyed by
``date, store_nbr, family``). This module regenerates that artifact from the
training sales so the variant runs from a clean checkout.

The per-``(store_nbr, family)`` cell floor is the low quantile (q10) of that
cell's historical daily sales, broadcast across every ``(date, store_nbr,
family)`` row. This encodes the weighting intent: high-volume cells get a high
floor and are down-weighted; sparse/low cells get a near-zero floor and are
up-weighted, focusing the loss on the harder low-volume series.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FLOOR_QUANTILE = 0.10
FLOOR_FILENAME = "floor_per_row.parquet"


def build_floor_frame(train: pd.DataFrame,
                      quantile: float = FLOOR_QUANTILE) -> pd.DataFrame:
    """Compute the per-cell floor table from training sales.

    Args:
        train: Long training frame with ``date, store_nbr, family, sales``.
        quantile: Low quantile of each cell's sales used as its floor.

    Returns:
        A frame with columns ``date, store_nbr, family, cell_floor`` covering
        every ``(date, store_nbr, family)`` row in ``train``.
    """
    cell_floor = (
        train.groupby(["store_nbr", "family"])["sales"]
        .quantile(quantile)
        .rename("cell_floor")
        .reset_index()
    )
    return (
        train[["date", "store_nbr", "family"]]
        .merge(cell_floor, on=["store_nbr", "family"], how="left")
    )


def ensure_floor_parquet(train: pd.DataFrame, data_dir: Path,
                         quantile: float = FLOOR_QUANTILE) -> Path:
    """Return the floor parquet path, regenerating it from ``train`` if absent.

    Args:
        train: Long training frame with ``date, store_nbr, family, sales``.
        data_dir: Directory that holds (or will hold) the parquet.
        quantile: Low quantile passed to :func:`build_floor_frame`.

    Returns:
        Path to ``floor_per_row.parquet`` (existing or freshly written).
    """
    path = data_dir / FLOOR_FILENAME
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = build_floor_frame(train, quantile)
    frame.to_parquet(path, index=False)
    print(
        f">>> floor_per_row.parquet absent — regenerated from training sales "
        f"(q{int(quantile * 100)} per store×family, rows={len(frame)}) -> {path}"
    )
    return path
