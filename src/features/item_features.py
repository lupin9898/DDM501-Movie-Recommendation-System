"""Item-level feature engineering from MovieLens ratings, movies, and tags data."""

import logging
import re
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_year(title: str) -> int | None:
    """Extract the four-digit release year from a movie title string.

    Expects the common MovieLens format, e.g. ``"Toy Story (1995)"``.
    Returns ``None`` when no year can be parsed.
    """
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    if match:
        return int(match.group(1))
    return None


def _multi_hot_genres(movies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a DataFrame of multi-hot genre columns and the list of genre names.

    If the DataFrame already contains binary genre columns they are reused;
    otherwise the pipe-delimited ``genres`` column is split.
    """
    if "genres" not in movies.columns:
        log.warning("No 'genres' column in movies DataFrame – skipping genre encoding")
        return pd.DataFrame(index=movies.index), []

    all_genres: set[str] = set()
    for entry in movies["genres"].dropna():
        all_genres.update(
            g.strip()
            for g in str(entry).split("|")
            if g.strip() and g.strip() != "(no genres listed)"
        )
    genre_list = sorted(all_genres)

    genre_df = pd.DataFrame(0, index=movies.index, columns=genre_list)
    for genre in genre_list:
        genre_df[genre] = movies["genres"].apply(lambda x, g=genre: int(g in str(x).split("|")))
    return genre_df, genre_list


# ---------------------------------------------------------------------------
# Tag TF-IDF
# ---------------------------------------------------------------------------


def build_tag_tfidf(
    tags: pd.DataFrame,
    movie_idx_col: str = "movie_idx",
    max_features: int = 100,
) -> scipy.sparse.csr_matrix:
    """Build a TF-IDF matrix from user-generated tags.

    Parameters
    ----------
    tags:
        Must contain ``movie_idx_col`` and ``tag`` columns.
    movie_idx_col:
        Column name used as the movie identifier.
    max_features:
        Maximum number of TF-IDF features to retain.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape ``(n_movies, max_features)`` where movies are
        ordered by ascending ``movie_idx_col``.  The row order matches the
        sorted unique movie indices in *tags*.
    """
    required = {movie_idx_col, "tag"}
    missing = required - set(tags.columns)
    if missing:
        raise ValueError(f"tags DataFrame is missing columns: {missing}")

    # Concatenate all tags per movie into a single document string.
    docs = (
        tags.dropna(subset=["tag"])
        .groupby(movie_idx_col)["tag"]
        .apply(lambda x: " ".join(x.astype(str)))
        .sort_index()
    )

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix: scipy.sparse.csr_matrix = vectorizer.fit_transform(docs.values)

    log.info(
        "Tag TF-IDF matrix: %d movies x %d features",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
    )
    return tfidf_matrix


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_item_features(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a feature DataFrame indexed by ``movie_idx``.

    Parameters
    ----------
    ratings:
        Must contain ``movie_idx`` and ``rating`` columns.
    movies:
        Must contain ``movie_idx``, ``title``, and a pipe-delimited ``genres``
        column.
    tags:
        Optional.  Must contain ``movie_idx`` and ``tag`` if provided.

    Returns
    -------
    pd.DataFrame
        One row per movie with columns:
        ``avg_rating``, ``num_ratings``, ``popularity_score``, ``year``,
        genre multi-hot columns (e.g. ``Action``, ``Comedy``, …), and
        optionally ``tag_tfidf_0`` … ``tag_tfidf_N`` when *tags* is supplied.
    """
    required_rating_cols = {"movie_idx", "rating"}
    missing = required_rating_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings DataFrame is missing columns: {missing}")

    required_movie_cols = {"movie_idx", "title"}
    missing_m = required_movie_cols - set(movies.columns)
    if missing_m:
        raise ValueError(f"movies DataFrame is missing columns: {missing_m}")

    log.info("Building item features for %d unique movies", movies["movie_idx"].nunique())

    # --- basic rating aggregation --------------------------------------------
    agg: pd.DataFrame = cast(
        pd.DataFrame,
        ratings.groupby("movie_idx")["rating"].agg(
            avg_rating="mean",
            num_ratings="count",
        ),
    )

    # Ensure every movie in *movies* is present, even without ratings.
    agg = agg.reindex(movies["movie_idx"].unique())
    agg["avg_rating"] = agg["avg_rating"].fillna(0.0)
    agg["num_ratings"] = agg["num_ratings"].fillna(0).astype(int)

    # --- popularity score (log of count) -------------------------------------
    agg["popularity_score"] = np.log1p(agg["num_ratings"])

    # --- release year --------------------------------------------------------
    year_map = movies.set_index("movie_idx")["title"].apply(_extract_year)
    agg["year"] = year_map.reindex(agg.index)

    # --- multi-hot genre encoding --------------------------------------------
    movies_indexed = movies.set_index("movie_idx")
    genre_df, genre_list = _multi_hot_genres(movies_indexed)
    if genre_list:
        genre_df = genre_df.reindex(agg.index, fill_value=0)
        agg = pd.concat([agg, genre_df], axis=1)

    # --- tag TF-IDF (optional) -----------------------------------------------
    if tags is not None and not tags.empty:
        tfidf_matrix = build_tag_tfidf(tags, movie_idx_col="movie_idx")

        # Map back to the correct movie indices.
        tfidf_indices = (
            tags.dropna(subset=["tag"])
            .groupby("movie_idx")["tag"]
            .apply(lambda x: " ".join(x.astype(str)))
            .sort_index()
            .index
        )
        n_features = tfidf_matrix.shape[1]
        tfidf_cols = [f"tag_tfidf_{i}" for i in range(n_features)]
        tfidf_dense = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=tfidf_indices,
            columns=tfidf_cols,
        )
        agg = agg.join(tfidf_dense, how="left")
        agg[tfidf_cols] = agg[tfidf_cols].fillna(0.0)

    agg.index.name = "movie_idx"
    log.info("Item feature matrix shape: %s", agg.shape)
    return agg
