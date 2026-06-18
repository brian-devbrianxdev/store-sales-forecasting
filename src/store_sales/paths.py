"""Single source of truth for project directory locations.

Every other module imports its paths from here instead of recomputing
``Path(__file__).resolve().parents[N]`` ad hoc. The repository root is the
directory three levels above this file (``src/store_sales/paths.py`` →
``<repo>``).

Attributes:
    ROOT: Absolute path to the repository root.
    DATA: Directory holding the raw competition CSVs (``train.csv`` etc.).
    SUBMISSIONS: Directory holding every leg's prediction CSV and the final blend.
    OUT: Alias of :data:`SUBMISSIONS`. The legacy ``neural_tsmixer.py`` wrote to a
        separate ``out/`` directory; the refactor unifies every leg onto
        ``submissions/`` so the blend reads a single location.
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data"
SUBMISSIONS: Path = ROOT / "submissions"
OUT: Path = SUBMISSIONS
CONFIG_FILE: Path = ROOT / "config.yaml"


def ensure_dirs() -> None:
    """Create the submissions directory if it does not yet exist.

    The raw :data:`DATA` directory is intentionally *not* created — a missing
    data directory is a hard error surfaced by the loaders, not something to
    paper over.
    """
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
