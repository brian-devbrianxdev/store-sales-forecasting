"""Raw competition-CSV loading.

Consolidates the per-script ``pd.read_csv(... parse_dates=["date"])`` blocks
that were duplicated across every leg. The dataframes are returned untouched
(no feature engineering) so each leg can apply its own preprocessing.
"""
from __future__ import annotations

import pandas as pd

from .. import paths

# Files keyed by the name each leg refers to them by.
_FILES: dict[str, str] = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}

# Files that carry a parseable ``date`` column.
_DATE_FILES = {"train", "test", "oil", "holidays", "transactions"}


def load_raw_frames() -> dict[str, pd.DataFrame]:
    """Read every raw competition CSV from :data:`store_sales.paths.DATA`.

    Returns:
        A dict mapping each logical name to its dataframe:
        ``{"train", "test", "stores", "oil", "holidays", "transactions"}``.
        Every frame except ``stores`` has its ``date`` column parsed to
        ``datetime64``.

    Raises:
        FileNotFoundError: If any expected CSV is missing from the data dir.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in _FILES.items():
        csv_path = paths.DATA / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"missing data file: {csv_path}")
        parse_dates = ["date"] if name in _DATE_FILES else None
        frames[name] = pd.read_csv(csv_path, parse_dates=parse_dates)
    return frames
