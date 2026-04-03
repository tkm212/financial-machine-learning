"""Load local datasets for Elements of Statistical Learning-style notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATPWTA_SUBDIR = "atpwta-tennis-data"
TMDB_SUBDIR = "tmdb-movie-metadata"

TARGET_PRIORITY = (
    "winner_rank_points",
    "minutes",
    "loser_rank_points",
)


def load_atpwta_regression(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Read ATP/WTA *match* CSVs from ``inputs_dir / atpwta-tennis-data`` (see
    ``scripts/download_atpwta_tennis_data.py``) and return numeric ``X``, continuous ``y``.

    Uses the first available target in order: ``winner_rank_points``, ``minutes``,
    ``loser_rank_points`` (common in Sackmann-style match files).
    """
    root = inputs_dir / ATPWTA_SUBDIR
    if not root.is_dir():
        msg = f"Missing {root}. Run: uv run python scripts/download_atpwta_tennis_data.py"
        raise FileNotFoundError(msg)

    paths = sorted(root.rglob("*.csv"))
    match_paths = [p for p in paths if "match" in p.name.lower()]
    use_paths = match_paths or paths
    if not use_paths:
        msg = f"No CSV files under {root}"
        raise FileNotFoundError(msg)

    frames: list[pd.DataFrame] = []
    for p in use_paths:
        try:
            frames.append(pd.read_csv(p, low_memory=False))
        except OSError, UnicodeDecodeError, pd.errors.EmptyDataError:
            continue

    if not frames:
        msg = f"Could not read any CSV from {root}"
        raise ValueError(msg)

    df = pd.concat(frames, ignore_index=True)

    target = next((c for c in TARGET_PRIORITY if c in df.columns), None)
    if target is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        msg = f"No target in {TARGET_PRIORITY}. Numeric columns (sample): {numeric_cols[:20]}"
        raise ValueError(msg)

    y_raw = df[target]
    X = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
    tbl = pd.concat([y_raw, X], axis=1).dropna()
    y = tbl[target]
    X = tbl.drop(columns=[target])

    if len(X) < 50:
        msg = f"Too few complete rows after dropna: {len(X)}"
        raise ValueError(msg)

    return X, y, target


def load_tmdb_revenue_regression(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Load TMDB movie CSV(s) from ``inputs_dir / tmdb-movie-metadata`` (see
    ``scripts/download_tmdb_movie_metadata.py``). Regress **revenue** on numeric
    features: ``budget``, ``popularity``, ``runtime``, ``vote_average``, ``vote_count``,
    and **release_year** parsed from ``release_date`` when present.

    Rows with missing or non-positive revenue/budget are dropped so the target is
    usable for log-scale or standard regression.
    """
    root = inputs_dir / TMDB_SUBDIR
    if not root.is_dir():
        msg = f"Missing {root}. Run: uv run python scripts/download_tmdb_movie_metadata.py"
        raise FileNotFoundError(msg)

    candidates = sorted(root.rglob("*.csv"))
    preferred = [p for p in candidates if p.name.lower() == "tmdb_5000_movies.csv"]
    if not preferred:
        preferred = [p for p in candidates if "movie" in p.name.lower() and "credit" not in p.name.lower()]
    if not preferred:
        preferred = candidates
    if not preferred:
        msg = f"No CSV files under {root}"
        raise FileNotFoundError(msg)

    path = preferred[0]
    df = pd.read_csv(path, low_memory=False)

    target = "revenue"
    if target not in df.columns:
        msg = f"Expected column {target!r} in {path.name}; got {list(df.columns)[:25]}"
        raise ValueError(msg)

    df = df.copy()
    for col in ("budget", "revenue"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    df = df[df["revenue"] > 0]
    df = df[df["budget"] > 0]

    y_raw = df[target]
    drop = {target, "id", "imdb_id"}
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    tbl = pd.concat([y_raw, X], axis=1).dropna()
    y = tbl[target]
    X = tbl.drop(columns=[target])

    if len(X) < 50:
        msg = f"Too few complete rows after filters: {len(X)}"
        raise ValueError(msg)

    return X, y, target
