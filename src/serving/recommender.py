"""Recommendation service that loads model artifacts and serves predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_RECOMMEND_CACHE_MAXSIZE = 1024


class RecommenderServiceError(Exception):
    """Raised when the recommender service encounters an error."""


class ModelNotLoadedError(RecommenderServiceError):
    """Raised when inference is attempted before loading model artifacts."""


class UnknownMovieError(RecommenderServiceError):
    """Raised when a movie ID is not found in the catalog."""


class RecommenderService:
    """Load ALS model artifacts and serve real-time recommendations.

    The service expects the following artifacts:
    - ``model.pkl``: pickled dict with ``user_factors`` and ``item_factors`` arrays
    - ``data_dir/movies.csv``: movie metadata (movieId, title, genres)
    - ``data_dir/user_mapping.csv``: external userId to internal user_idx
    - ``data_dir/item_mapping.csv``: external movieId to internal item_idx
    - ``data_dir/user_seen_items.pkl``: dict mapping user_idx to set of item_idx
    """

    def __init__(self) -> None:
        self._loaded: bool = False
        self._user_factors: NDArray[Any] = np.array([])
        self._item_factors: NDArray[Any] = np.array([])
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._movie_meta: dict[int, dict[str, str | list[str]]] = {}
        self._user_seen: dict[int, set[int]] = {}
        self._popular_items: list[dict[str, int | str | float | list[str]]] = []
        self._model_meta: dict[str, Any] = {}
        self._recommend_cache: dict[
            tuple[int, int, bool], list[dict[str, int | str | float | list[str]]]
        ] = {}

    @property
    def is_loaded(self) -> bool:
        """Whether model artifacts have been loaded."""
        return self._loaded

    @property
    def model_meta(self) -> dict[str, Any]:
        """Training metadata from the last loaded model (trained_at, run_id, …)."""
        return dict(self._model_meta)

    def load(self, model_path: Path, data_dir: Path) -> None:
        """Load model artifacts, movie metadata, user/item mappings, and user seen items.

        Parameters
        ----------
        model_path:
            Path to the pickled model file containing ``user_factors`` and
            ``item_factors``.
        data_dir:
            Directory containing ``movies.csv``, ``user_mapping.csv``,
            ``item_mapping.csv``, and ``user_seen_items.pkl``.
        """
        log.info("Loading model from %s", model_path)
        if not model_path.exists():
            raise RecommenderServiceError(f"Model file not found: {model_path}")

        # Saved by joblib.dump() in train.py
        model_data: dict[str, NDArray[Any]] = joblib.load(model_path)

        self._user_factors = model_data["user_factors"]
        self._item_factors = model_data["item_factors"]
        log.info(
            "Loaded factors — users: %s, items: %s",
            self._user_factors.shape,
            self._item_factors.shape,
        )

        # --- user mapping (user_id_map.parquet: original_id → encoded_id) --------
        user_map_path = data_dir / "user_id_map.parquet"
        if user_map_path.exists():
            user_map_df = pd.read_parquet(user_map_path)
            self._user_to_idx = dict(
                zip(
                    user_map_df["original_id"].astype(int),
                    user_map_df["encoded_id"].astype(int),
                    strict=True,
                )
            )
            log.info("Loaded user mapping: %d users", len(self._user_to_idx))
        else:
            log.warning("User mapping not found at %s — assuming identity mapping", user_map_path)
            self._user_to_idx = {i: i for i in range(self._user_factors.shape[0])}

        # --- item mapping (movie_id_map.parquet: original_id → encoded_id) -------
        item_map_path = data_dir / "movie_id_map.parquet"
        if item_map_path.exists():
            item_map_df = pd.read_parquet(item_map_path)
            self._item_to_idx = dict(
                zip(
                    item_map_df["original_id"].astype(int),
                    item_map_df["encoded_id"].astype(int),
                    strict=True,
                )
            )
            self._idx_to_item = {v: k for k, v in self._item_to_idx.items()}
            log.info("Loaded item mapping: %d items", len(self._item_to_idx))
        else:
            log.warning("Item mapping not found at %s — assuming identity mapping", item_map_path)
            self._item_to_idx = {i: i for i in range(self._item_factors.shape[0])}
            self._idx_to_item = {i: i for i in range(self._item_factors.shape[0])}

        # --- movie metadata (movies.parquet: movieId, title, genres) -------------
        movies_path = data_dir / "movies.parquet"
        if movies_path.exists():
            movies_df = pd.read_parquet(movies_path)
            for _, row in movies_df.iterrows():
                mid = int(row["movieId"])
                title = str(row.get("title", f"Movie {mid}"))
                raw_genres = str(row.get("genres", ""))
                genres = [
                    g.strip()
                    for g in raw_genres.split("|")
                    if g.strip() and g.strip() != "(no genres listed)"
                ]
                self._movie_meta[mid] = {"title": title, "genres": genres}
            log.info("Loaded movie metadata: %d movies", len(self._movie_meta))
        else:
            log.warning("Movies metadata not found at %s", movies_path)

        # --- user seen items (computed from train.parquet) -----------------------
        # Prefer joblib (sandboxed loader, matches how the model was serialized)
        # so we don't fall back to unsafe stdlib pickle for this artifact.
        seen_path = data_dir / "user_seen_items.joblib"
        legacy_seen_path = data_dir / "user_seen_items.pkl"
        if seen_path.exists():
            self._user_seen = joblib.load(seen_path)
            log.info("Loaded user seen items for %d users", len(self._user_seen))
        elif legacy_seen_path.exists():
            self._user_seen = joblib.load(legacy_seen_path)
            log.info(
                "Loaded user seen items from legacy .pkl for %d users — migrate to .joblib",
                len(self._user_seen),
            )
        else:
            train_path = data_dir / "train.parquet"
            if train_path.exists():
                train_df = pd.read_parquet(train_path, columns=["user_idx", "movie_idx"])
                grouped: dict[Any, set[Any]] = (
                    train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
                )
                self._user_seen = {int(k): {int(x) for x in v} for k, v in grouped.items()}
                log.info("Built user seen items from train.parquet: %d users", len(self._user_seen))
            else:
                log.warning(
                    "User seen items not found at %s — exclude_seen will have no effect", seen_path
                )

        # --- pre-compute popular items -------------------------------------------
        self._popular_items = self._compute_popular_items(top_k=100)
        log.info("Pre-computed top-%d popular items", len(self._popular_items))

        # --- model metadata (trained_at, run_id, metrics) — written by train.py --
        meta_path = model_path.parent / "model_meta.json"
        if meta_path.exists():
            try:
                self._model_meta = json.loads(meta_path.read_text())
                log.info(
                    "Loaded model metadata: trained_at=%s run_id=%s",
                    self._model_meta.get("trained_at"),
                    self._model_meta.get("mlflow_run_id"),
                )
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Failed to parse %s: %s", meta_path, exc)
                self._model_meta = {}
        else:
            log.info("No model_meta.json next to model — metadata will be empty")
            self._model_meta = {}

        self._loaded = True
        log.info("RecommenderService ready")

    def _ensure_loaded(self) -> None:
        """Raise if model artifacts have not been loaded."""
        if not self._loaded:
            raise ModelNotLoadedError("Model artifacts have not been loaded. Call load() first.")

    def is_known_user(self, user_id: int) -> bool:
        """Return True if user_id is in the trained user mapping."""
        return user_id in self._user_to_idx

    def _compute_popular_items(
        self, top_k: int = 100
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Compute popular items by counting how many users have seen each item."""
        item_counts: dict[int, int] = {}
        for seen_set in self._user_seen.values():
            for item_idx in seen_set:
                item_counts[item_idx] = item_counts.get(item_idx, 0) + 1

        if not item_counts:
            # Fallback: use item factor norms as a proxy for popularity
            norms = np.linalg.norm(self._item_factors, axis=1)
            ranked_indices = np.argsort(-norms)[:top_k]
            results: list[dict[str, int | str | float | list[str]]] = []
            for idx in ranked_indices:
                movie_id = self._idx_to_item.get(int(idx), int(idx))
                meta = self._movie_meta.get(movie_id, {})
                results.append(
                    {
                        "movie_id": movie_id,
                        "title": str(meta.get("title", f"Movie {movie_id}")),
                        "score": float(norms[idx]),
                        "genres": list(meta.get("genres", [])),
                    }
                )
            return results

        max_count = max(item_counts.values())
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for item_idx, count in sorted_items:
            movie_id = self._idx_to_item.get(item_idx, item_idx)
            meta = self._movie_meta.get(movie_id, {})
            results.append(
                {
                    "movie_id": movie_id,
                    "title": str(meta.get("title", f"Movie {movie_id}")),
                    "score": round(float(count) / max_count, 6),
                    "genres": list(meta.get("genres", [])),
                }
            )
        return results

    def _format_item(self, item_idx: int, score: float) -> dict[str, int | str | float | list[str]]:
        """Build a recommendation dict from an internal item index and score."""
        movie_id = self._idx_to_item.get(item_idx, item_idx)
        meta = self._movie_meta.get(movie_id, {})
        return {
            "movie_id": movie_id,
            "title": str(meta.get("title", f"Movie {movie_id}")),
            "score": round(float(score), 6),
            "genres": list(meta.get("genres", [])),
        }

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Generate recommendations for a user.

        Returns a list of dicts with keys: movie_id, title, score, genres.
        Falls back to popular items for unknown users (cold start).
        """
        self._ensure_loaded()

        cache_key = (user_id, top_k, exclude_seen)
        if cache_key in self._recommend_cache:
            return [dict(item) for item in self._recommend_cache[cache_key]]

        user_idx = self._user_to_idx.get(user_id)

        # Cold start: unknown user -> return popular items
        if user_idx is None:
            log.warning("Unknown user_id=%d — returning popular items (cold start)", user_id)
            return [dict(item) for item in self._popular_items[:top_k]]

        # Compute scores via dot product: user_factors[uid] @ item_factors.T
        # .copy() makes the mutation below explicit and safe regardless of numpy internals.
        scores: NDArray[Any] = (self._user_factors[user_idx] @ self._item_factors.T).copy()

        # Exclude already-seen items by setting their scores to -inf
        if exclude_seen:
            seen = self._user_seen.get(user_idx, set())
            if seen:
                seen_indices = np.array(list(seen), dtype=np.intp)
                valid_mask = seen_indices < len(scores)
                scores[seen_indices[valid_mask]] = -np.inf

        # Get top_k indices via argpartition for efficiency
        if top_k < len(scores):
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            top_indices = np.argsort(-scores)[:top_k]

        result = [self._format_item(int(idx), scores[idx]) for idx in top_indices]

        if len(self._recommend_cache) >= _RECOMMEND_CACHE_MAXSIZE:
            self._recommend_cache.pop(next(iter(self._recommend_cache)))
        self._recommend_cache[cache_key] = result
        return list(result)

    def similar_items(
        self, movie_id: int, top_k: int = 10
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Find similar movies using item factor cosine similarity.

        Raises ``UnknownMovieError`` if the movie ID is not in the catalog.
        """
        self._ensure_loaded()

        item_idx = self._item_to_idx.get(movie_id)
        if item_idx is None:
            raise UnknownMovieError(f"Movie ID {movie_id} not found in catalog")

        # Cosine similarity: normalize then dot product
        target = self._item_factors[item_idx]
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            return []

        norms = np.linalg.norm(self._item_factors, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)

        similarities: NDArray[Any] = ((self._item_factors @ target) / (norms * target_norm)).copy()

        # Exclude the item itself
        similarities[item_idx] = -np.inf

        # Get top_k most similar
        if top_k < len(similarities):
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)[:top_k]

        return [self._format_item(int(idx), similarities[idx]) for idx in top_indices]

    def get_popular_items(self, top_k: int = 10) -> list[dict[str, int | str | float | list[str]]]:
        """Return most popular items as a fallback."""
        self._ensure_loaded()
        return [dict(item) for item in self._popular_items[:top_k]]
