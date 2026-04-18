"""User-level feature engineering from MovieLens ratings and movies data."""

import logging
from typing import cast

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Genre constants
# ---------------------------------------------------------------------------
MOVIELENS_GENRES: list[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def _extract_genre_columns(movies: pd.DataFrame) -> list[str]:
    """Return genre columns present in *movies*, creating them if needed.

    If the DataFrame already contains one-hot genre columns (e.g. ``Action``,
    ``Comedy``, …) they are reused.  Otherwise the function splits the
    pipe-delimited ``genres`` column and creates binary indicator columns.
    """
    existing = [g for g in MOVIELENS_GENRES if g in movies.columns]
    if existing:
        return existing

    if "genres" not in movies.columns:
        log.warning("No 'genres' column found in movies DataFrame – skipping genre features")
        return []

    all_genres: set[str] = set()
    for entry in movies["genres"].dropna():
        all_genres.update(
            g.strip() for g in str(entry).split("|") if g.strip() != "(no genres listed)"
        )

    genre_list = sorted(all_genres)
    for genre in genre_list:
        movies[genre] = movies["genres"].apply(lambda x, g=genre: int(g in str(x).split("|")))
    log.info("Created %d binary genre columns from pipe-delimited 'genres' field", len(genre_list))
    return genre_list


def _compute_genre_preference(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    genre_cols: list[str],
) -> pd.DataFrame:
    """Compute per-user genre proportion vector.

    For each user, the proportion is the fraction of their total watched movies
    belonging to each genre (a movie can belong to multiple genres, so
    proportions may sum to > 1 per user).
    """
    merged = ratings[["user_idx", "movie_idx"]].merge(
        movies[["movie_idx"] + genre_cols],
        on="movie_idx",
        how="left",
    )
    genre_sums = merged.groupby("user_idx")[genre_cols].sum()
    total_per_user = genre_sums.sum(axis=1).replace(0, 1)  # avoid division by zero
    genre_props = genre_sums.div(total_per_user, axis=0)
    genre_props.columns = [f"genre_pref_{g}" for g in genre_cols]
    return genre_props


def build_user_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Build a feature DataFrame indexed by ``user_idx``.

    Parameters
    ----------
    ratings:
        Must contain columns ``user_idx``, ``movie_idx``, ``rating``, and
        ``timestamp``.
    movies:
        Must contain ``movie_idx`` and either pre-encoded genre columns or a
        pipe-delimited ``genres`` column.

    Returns
    -------
    pd.DataFrame
        One row per user with columns:
        ``avg_rating``, ``num_ratings``, ``rating_std``,
        ``genre_pref_<Genre>`` (one per genre), ``activity_recency``.
    """
    required_rating_cols = {"user_idx", "movie_idx", "rating", "timestamp"}
    missing = required_rating_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings DataFrame is missing columns: {missing}")

    log.info("Building user features for %d unique users", ratings["user_idx"].nunique())

    # --- basic aggregation ---------------------------------------------------
    agg: pd.DataFrame = cast(
        pd.DataFrame,
        ratings.groupby("user_idx")["rating"].agg(
            avg_rating="mean",
            num_ratings="count",
            rating_std="std",
        ),
    )
    # Single-rating users get NaN std; fill with 0.
    agg["rating_std"] = agg["rating_std"].fillna(0.0)

    # --- genre preference ----------------------------------------------------
    genre_cols = _extract_genre_columns(movies)
    if genre_cols:
        genre_pref = _compute_genre_preference(ratings, movies, genre_cols)
        agg = agg.join(genre_pref, how="left")
        # Users with no genre info get 0
        genre_pref_cols = [c for c in agg.columns if c.startswith("genre_pref_")]
        agg[genre_pref_cols] = agg[genre_pref_cols].fillna(0.0)

    # --- activity recency (seconds since last interaction) --------------------
    max_ts = ratings["timestamp"].max()
    last_ts = ratings.groupby("user_idx")["timestamp"].max()
    agg["activity_recency"] = (max_ts - last_ts).astype(np.float64)

    log.info("User feature matrix shape: %s", agg.shape)
    return agg
