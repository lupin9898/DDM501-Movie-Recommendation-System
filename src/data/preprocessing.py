"""Load raw CSVs, filter, encode IDs, temporal split, and save as parquet."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PreprocessingError(Exception):
    """Raised when any preprocessing step fails."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    """Summary statistics for a train/val/test split."""

    train_size: int
    val_size: int
    test_size: int
    n_users: int
    n_items: int
    sparsity: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_csv(raw_dir: Path, filename: str) -> pd.DataFrame:
    """Load a single CSV from *raw_dir* and return a DataFrame."""
    path = raw_dir / filename
    if not path.exists():
        raise PreprocessingError(f"Raw file not found: {path}")
    log.info("Loading %s ...", path)
    df = pd.read_csv(path)
    log.info("  %d rows, %d columns", len(df), len(df.columns))
    return df


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ratings, movies, and optionally tags CSVs from *raw_dir*.

    ``tags.csv`` is optional — if absent, an empty DataFrame with the expected
    columns is returned instead.
    """
    ratings = load_raw_csv(raw_dir, "ratings.csv")
    movies = load_raw_csv(raw_dir, "movies.csv")

    tags_path = raw_dir / "tags.csv"
    if tags_path.exists():
        tags = load_raw_csv(raw_dir, "tags.csv")
    else:
        log.info("tags.csv not found — continuing without tags")
        tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    return ratings, movies, tags


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_cold_start(
    ratings: pd.DataFrame,
    min_user_ratings: int,
    min_item_ratings: int,
) -> pd.DataFrame:
    """Remove users with < *min_user_ratings* and movies with < *min_item_ratings*.

    Filtering is applied iteratively until convergence, because removing
    items may cause some users to drop below the threshold and vice-versa.
    """
    prev_len = -1
    current = ratings.copy()
    iteration = 0

    while len(current) != prev_len:
        prev_len = len(current)
        iteration += 1

        user_counts = current["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        current = current[current["userId"].isin(valid_users)]

        item_counts = current["movieId"].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        current = current[current["movieId"].isin(valid_items)]

        log.info(
            "  filter iteration %d: %d rows (removed %d)",
            iteration,
            len(current),
            prev_len - len(current),
        )

    log.info(
        "Cold-start filter done — %d → %d ratings (%d removed)",
        len(ratings),
        len(current),
        len(ratings) - len(current),
    )
    return current.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ID encoding
# ---------------------------------------------------------------------------


def build_id_mapping(series: pd.Series) -> pd.DataFrame:  # type: ignore[type-arg]
    """Create a contiguous 0-based mapping for unique values in *series*.

    Returns a two-column DataFrame: ``original_id`` and ``encoded_id``.
    """
    unique_ids = np.sort(series.unique())
    mapping = pd.DataFrame(
        {
            "original_id": unique_ids,
            "encoded_id": np.arange(len(unique_ids), dtype=np.int32),
        }
    )
    return mapping


def encode_ids(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace userId and movieId with contiguous 0-based integers.

    Returns ``(encoded_ratings, user_id_map, movie_id_map)``.
    """
    user_map = build_id_mapping(ratings["userId"])
    movie_map = build_id_mapping(ratings["movieId"])

    user_lookup = dict(zip(user_map["original_id"], user_map["encoded_id"], strict=True))
    movie_lookup = dict(zip(movie_map["original_id"], movie_map["encoded_id"], strict=True))

    encoded = ratings.copy()
    encoded["user_idx"] = encoded["userId"].map(user_lookup).astype(np.int32)
    encoded["movie_idx"] = encoded["movieId"].map(movie_lookup).astype(np.int32)
    encoded = encoded.drop(columns=["userId", "movieId"])

    log.info(
        "Encoded %d unique users, %d unique movies",
        len(user_map),
        len(movie_map),
    )
    return encoded, user_map, movie_map


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    ratings: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings by timestamp into train / val / test sets.

    The ``train_frac`` earliest interactions go to train, the next
    ``val_frac`` to validation, and the remainder to test.
    """
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0):
        raise PreprocessingError("Fractions must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise PreprocessingError("train_frac + val_frac must be < 1.0")

    sorted_df = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = sorted_df.iloc[:train_end].reset_index(drop=True)
    val = sorted_df.iloc[train_end:val_end].reset_index(drop=True)
    test = sorted_df.iloc[val_end:].reset_index(drop=True)

    log.info(
        "Temporal split → train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train),
        100 * len(train) / n,
        len(val),
        100 * len(val) / n,
        len(test),
        100 * len(test) / n,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save *df* to a parquet file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    log.info("Saved %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_sparsity(n_ratings: int, n_users: int, n_items: int) -> float:
    """Return the sparsity ratio of the interaction matrix."""
    if n_users == 0 or n_items == 0:
        return 1.0
    return 1.0 - n_ratings / (n_users * n_items)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def preprocess(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    min_user_ratings: int | None = None,
    min_item_ratings: int | None = None,
) -> SplitStats:
    """Run the full preprocessing pipeline: load ➜ filter ➜ encode ➜ split ➜ save.

    Parameters default to values from :pydata:`src.config.settings`.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    processed_dir = processed_dir or settings.data_processed_dir
    min_user_ratings = min_user_ratings if min_user_ratings is not None else settings.min_user_ratings
    min_item_ratings = min_item_ratings if min_item_ratings is not None else settings.min_item_ratings

    # 1. Load
    ratings, movies, tags = load_raw_data(raw_dir)

    # 2. Filter cold-start users/items
    ratings_filtered = filter_cold_start(ratings, min_user_ratings, min_item_ratings)

    # 3. Encode IDs to contiguous integers
    ratings_encoded, user_id_map, movie_id_map = encode_ids(ratings_filtered)

    # 4. Temporal split
    train, val, test = temporal_split(ratings_encoded)

    # 5. Save processed data
    save_parquet(train, processed_dir / "train.parquet")
    save_parquet(val, processed_dir / "val.parquet")
    save_parquet(test, processed_dir / "test.parquet")
    save_parquet(user_id_map, processed_dir / "user_id_map.parquet")
    save_parquet(movie_id_map, processed_dir / "movie_id_map.parquet")

    # 6. Also save filtered movies and tags (only movies that survived filtering)
    surviving_movie_ids = set(movie_id_map["original_id"])
    movies_filtered = movies[movies["movieId"].isin(surviving_movie_ids)].reset_index(drop=True)
    # Merge movie_idx so feature modules can join on it directly
    movies_filtered = movies_filtered.merge(
        movie_id_map.rename(columns={"original_id": "movieId", "encoded_id": "movie_idx"}),
        on="movieId",
        how="left",
    )
    save_parquet(movies_filtered, processed_dir / "movies.parquet")

    surviving_user_ids = set(user_id_map["original_id"])
    tags_filtered = tags[
        tags["movieId"].isin(surviving_movie_ids) & tags["userId"].isin(surviving_user_ids)
    ].reset_index(drop=True)
    save_parquet(tags_filtered, processed_dir / "tags.parquet")

    # 7. Compute and log summary statistics
    n_users = int(user_id_map["encoded_id"].nunique())
    n_items = int(movie_id_map["encoded_id"].nunique())
    total_ratings = len(train) + len(val) + len(test)
    sparsity = compute_sparsity(total_ratings, n_users, n_items)

    stats = SplitStats(
        train_size=len(train),
        val_size=len(val),
        test_size=len(test),
        n_users=n_users,
        n_items=n_items,
        sparsity=sparsity,
    )
    log.info("Preprocessing complete — %s", stats)
    log.info(
        "Sparsity: %.4f%% (%.2f%% of the user-item matrix is empty)",
        sparsity * 100,
        sparsity * 100,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point for ``python -m src.data.preprocessing``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    preprocess()


if __name__ == "__main__":
    main()
