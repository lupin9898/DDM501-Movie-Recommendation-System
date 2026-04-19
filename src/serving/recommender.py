"""Recommendation service — load LightFM artifact và phục vụ dự đoán realtime."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightfm import LightFM
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

_RECOMMEND_CACHE_MAXSIZE = 1024


class RecommenderServiceError(Exception):
    """Lỗi chung của recommender service."""


class ModelNotLoadedError(RecommenderServiceError):
    """Gọi inference trước khi load artifact."""


class UnknownMovieError(RecommenderServiceError):
    """movie_id không có trong catalog."""


class RecommenderService:
    """Load artifact LightFM và expose ``recommend`` / ``similar_items``.

    Artifact (``model.pkl``) do ``src/training/train.py`` dump ra, chứa:
    - ``model``: lightfm.LightFM đã fit.
    - ``item_features``: csr_matrix (n_items, n_features) — bắt buộc cho predict.
    - ``item_embeddings``: NDArray — cache cho similar_items.
    - ``user_id_map``: original userId -> internal idx.
    - ``item_id_map``: original movieId -> internal idx.
    - ``reverse_item_id_map``: internal idx -> original movieId.
    - ``user_seen``: dict user_idx -> set(item_idx) đã xem.
    - ``movies``: DataFrame movieId/title/genres (meta catalog).
    """

    def __init__(self) -> None:
        self._loaded: bool = False
        self._model: LightFM | None = None
        self._item_features: csr_matrix | None = None
        self._item_embeddings: NDArray[Any] = np.array([])
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._movie_meta: dict[int, dict[str, str | list[str]]] = {}
        self._user_seen: dict[int, set[int]] = {}
        self._popular_items: list[dict[str, int | str | float | list[str]]] = []
        self._model_meta: dict[str, Any] = {}
        self._num_threads: int = 4
        self._n_items: int = 0
        self._recommend_cache: dict[
            tuple[int, int, bool], list[dict[str, int | str | float | list[str]]]
        ] = {}

    @property
    def is_loaded(self) -> bool:
        """Artifact đã được load chưa."""
        return self._loaded

    @property
    def model_meta(self) -> dict[str, Any]:
        """Metadata training (trained_at, run_id, metrics)."""
        return dict(self._model_meta)

    # ------------------------------------------------------------------ load
    def load(self, model_path: Path, data_dir: Path) -> None:  # noqa: ARG002
        """Load artifact LightFM.

        ``data_dir`` giữ lại trong signature cho tương thích ngược nhưng không
        bắt buộc dùng — mọi mapping đều nằm trong artifact.
        """
        log.info("Loading artifact từ %s", model_path)
        if not model_path.exists():
            raise RecommenderServiceError(f"Model file not found: {model_path}")

        artifact: dict[str, Any] = joblib.load(model_path)
        model_type = artifact.get("model_type", "unknown")
        if model_type != "lightfm":
            raise RecommenderServiceError(
                f"Artifact model_type='{model_type}' không được serving hỗ trợ. "
                "Cần retrain bằng `python -m src.training.train --model lightfm`."
            )

        self._model = artifact["model"]
        self._item_features = artifact["item_features"]
        self._item_embeddings = np.asarray(artifact["item_embeddings"])
        self._num_threads = int(artifact.get("num_threads", 4))

        # Mapping — khóa JSON-friendly (int).
        self._user_to_idx = {int(k): int(v) for k, v in artifact["user_id_map"].items()}
        self._item_to_idx = {int(k): int(v) for k, v in artifact["item_id_map"].items()}
        self._idx_to_item = {int(k): int(v) for k, v in artifact["reverse_item_id_map"].items()}
        self._user_seen = {int(k): set(v) for k, v in artifact["user_seen"].items()}
        self._n_items = int(artifact["n_items"])

        log.info(
            "Loaded LightFM: users=%d, items=%d, item_features=%s",
            int(artifact["n_users"]),
            self._n_items,
            None if self._item_features is None else self._item_features.shape,
        )

        # --- movie metadata (tiêu đề + genres) ---
        movies_df: pd.DataFrame | None = artifact.get("movies")
        if movies_df is None:
            movies_path = data_dir / "movies.parquet"
            if movies_path.exists():
                movies_df = pd.read_parquet(movies_path)

        if movies_df is not None:
            self._movie_meta = {}
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
            log.info("Loaded movie metadata: %d phim", len(self._movie_meta))
        else:
            log.warning("Không tìm thấy movie metadata — title/genres sẽ bị trống")

        # Popular items (fallback cold-start) tính từ user_seen.
        self._popular_items = self._compute_popular_items(top_k=100)

        # model_meta.json — phục vụ /health.
        meta_path = model_path.parent / "model_meta.json"
        if meta_path.exists():
            try:
                self._model_meta = json.loads(meta_path.read_text())
                log.info(
                    "model_meta.json: trained_at=%s run_id=%s",
                    self._model_meta.get("trained_at"),
                    self._model_meta.get("mlflow_run_id"),
                )
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Không parse được %s: %s", meta_path, exc)
                self._model_meta = {}

        self._loaded = True
        log.info("RecommenderService sẵn sàng")

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise ModelNotLoadedError("Chưa load artifact. Gọi load() trước.")

    def is_known_user(self, user_id: int) -> bool:
        """Kiểm tra user có trong mapping training không."""
        return user_id in self._user_to_idx

    # ---------------------------------------------------------- popularity
    def _compute_popular_items(
        self, top_k: int = 100
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Đếm item đã được nhiều user xem nhất — fallback cho cold-start."""
        counts: dict[int, int] = {}
        for seen_set in self._user_seen.values():
            for item_idx in seen_set:
                counts[item_idx] = counts.get(item_idx, 0) + 1

        if not counts:
            return []

        max_count = max(counts.values())
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[dict[str, int | str | float | list[str]]] = []
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
        movie_id = self._idx_to_item.get(item_idx, item_idx)
        meta = self._movie_meta.get(movie_id, {})
        return {
            "movie_id": movie_id,
            "title": str(meta.get("title", f"Movie {movie_id}")),
            "score": round(float(score), 6),
            "genres": list(meta.get("genres", [])),
        }

    # ---------------------------------------------------------- recommend
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Gợi ý top-k phim cho user. Cold-start → trả về popular items."""
        self._ensure_loaded()
        assert self._model is not None  # noqa: S101  (checked by _ensure_loaded)

        cache_key = (user_id, top_k, exclude_seen)
        if cache_key in self._recommend_cache:
            return [dict(item) for item in self._recommend_cache[cache_key]]

        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            log.warning("User %d không có trong training — fallback popular", user_id)
            return [dict(item) for item in self._popular_items[:top_k]]

        item_ids = np.arange(self._n_items, dtype=np.int32)
        scores: NDArray[Any] = self._model.predict(
            user_ids=user_idx,
            item_ids=item_ids,
            item_features=self._item_features,
            num_threads=self._num_threads,
        ).astype(np.float64, copy=True)

        if exclude_seen:
            seen = self._user_seen.get(user_idx, set())
            if seen:
                seen_idx = np.fromiter(seen, dtype=np.intp)
                valid = seen_idx < len(scores)
                scores[seen_idx[valid]] = -np.inf

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

    # Alias theo yêu cầu người dùng (recommend_movies).
    def recommend_movies(
        self, user_id: int, num_recommendations: int = 10
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Alias Việt hoá-friendly cho ``recommend``.

        Trả về list dict: ``{movie_id, title, score, genres}``.
        """
        return self.recommend(user_id=user_id, top_k=num_recommendations, exclude_seen=True)

    # --------------------------------------------------------- similar_items
    def similar_items(
        self, movie_id: int, top_k: int = 10
    ) -> list[dict[str, int | str | float | list[str]]]:
        """Tìm phim tương tự dựa trên cosine similarity của item embedding."""
        self._ensure_loaded()

        item_idx = self._item_to_idx.get(movie_id)
        if item_idx is None:
            raise UnknownMovieError(f"Movie ID {movie_id} not found in catalog")

        target = self._item_embeddings[item_idx]
        target_norm = float(np.linalg.norm(target))
        if target_norm == 0.0:
            return []

        norms = np.linalg.norm(self._item_embeddings, axis=1)
        norms = np.where(norms == 0.0, 1.0, norms)

        sims: NDArray[Any] = ((self._item_embeddings @ target) / (norms * target_norm)).astype(
            np.float64, copy=True
        )
        sims[item_idx] = -np.inf

        if top_k < len(sims):
            top_indices = np.argpartition(-sims, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
        else:
            top_indices = np.argsort(-sims)[:top_k]
        return [self._format_item(int(idx), sims[idx]) for idx in top_indices]

    def get_popular_items(self, top_k: int = 10) -> list[dict[str, int | str | float | list[str]]]:
        """Trả popular items làm fallback."""
        self._ensure_loaded()
        return [dict(item) for item in self._popular_items[:top_k]]
