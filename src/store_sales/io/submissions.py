"""Reading and writing Kaggle submission CSVs.

A submission is a two-column ``(id, sales)`` frame. The ensemble works in
``log1p`` space, so :func:`load_log` is the canonical reader used by the blend;
:func:`write_submission` is the shared writer used by every leg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths


def load_log(file: str) -> np.ndarray:
    """Read a submission CSV (id-sorted) and return ``log1p`` of its sales.

    Sales are clipped to be non-negative before the log transform. The rows are
    sorted by ``id`` so the returned vector aligns across every leg's file.

    Args:
        file: Filename inside :data:`store_sales.paths.SUBMISSIONS`.

    Returns:
        ``log1p(sales)`` as a 1-D array, shape ``(n_rows,)``.
    """
    sales = (
        pd.read_csv(paths.SUBMISSIONS / file)
        .sort_values("id")
        .reset_index(drop=True)
        .sales.clip(lower=0)
        .to_numpy()
    )
    return np.log1p(sales)


def canonical_ids(file: str) -> pd.Series:
    """Return the sorted ``id`` column shared by every submission file.

    Args:
        file: Any submission filename inside the submissions directory.

    Returns:
        The ``id`` column, sorted ascending, index reset.
    """
    return (
        pd.read_csv(paths.SUBMISSIONS / file)
        .sort_values("id")
        .reset_index(drop=True)["id"]
    )


def write_submission(frame: pd.DataFrame, filename: str) -> Path:
    """Write an ``(id, sales)`` frame to the submissions directory.

    The frame is sorted by ``id`` before writing so output ordering is
    deterministic regardless of how predictions were assembled.

    Args:
        frame: Must contain ``id`` and ``sales`` columns.
        filename: Output filename (written into the submissions directory).

    Returns:
        The path that was written.
    """
    paths.ensure_dirs()
    out_path = paths.SUBMISSIONS / filename
    frame[["id", "sales"]].sort_values("id").to_csv(out_path, index=False)
    return out_path
