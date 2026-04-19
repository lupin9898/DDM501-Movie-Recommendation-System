"""Hybrid Recommender dùng LightFM (WARP loss).

Model này kế thừa ``BaseRecommender`` để giữ nguyên interface với các phần
khác của hệ thống (training pipeline, serving layer, evaluation).

Khác biệt quan trọng so với ALS cũ:
- ``predict`` của LightFM cần cả ``item_features`` ở thời điểm inference, nên
  model lưu lại ``self._item_features`` sau khi fit.
- ``similar_items`` tính qua embedding + cosine thay vì gọi implicit API.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from lightfm import LightFM
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from src.models.baseline import BaseRecommender

log = logging.getLogger(__name__)


class LightFMRecommender(BaseRecommender):
    """Hybrid matrix-factorization model với user-item interactions + item features."""

    def __init__(
        self,
        no_components: int = 64,
        loss: str = "warp",
        learning_rate: float = 0.05,
        epochs: int = 30,
        num_threads: int = 4,
        random_state: int = 42,
    ) -> None:
        self._no_components = no_components
        self._loss = loss
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._num_threads = num_threads
        self._random_state = random_state

        self._model: LightFM | None = None
        self._item_features: csr_matrix | None = None
        self._interaction: csr_matrix | None = None
        self._user_seen: dict[int, set[int]] = {}
        # Cache embedding item để phục vụ similar_items mà không phải gọi
        # get_item_representations() lặp lại mỗi request.
        self._item_embeddings: NDArray[Any] | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> None:
        """Huấn luyện LightFM trên matrix tương tác + item features (nếu có).

        Keyword arguments:
            item_features: ``csr_matrix`` (n_items, n_features) — bắt buộc để
                tận dụng kiến trúc hybrid. Nếu để ``None`` thì LightFM sẽ chạy
                ở chế độ collaborative-only.
        """
        item_features = kwargs.get("item_features")
        self._item_features = item_features
        self._interaction = interaction_matrix

        # Ghi nhận items mỗi user đã xem để ``exclude_seen`` hoạt động.
        csr = interaction_matrix.tocsr()
        self._user_seen = {}
        for uid in range(csr.shape[0]):
            start, end = int(csr.indptr[uid]), int(csr.indptr[uid + 1])
            self._user_seen[uid] = {int(x) for x in csr.indices[start:end]}

        self._model = LightFM(
            no_components=self._no_components,
            loss=self._loss,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
        )

        log.info(
            "Training LightFM: loss=%s, no_components=%d, lr=%.4f, epochs=%d, threads=%d",
            self._loss,
            self._no_components,
            self._learning_rate,
            self._epochs,
            self._num_threads,
        )
        self._model.fit(
            interactions=interaction_matrix,
            item_features=item_features,
            epochs=self._epochs,
            num_threads=self._num_threads,
            verbose=False,
        )

        # Tính và cache embedding item (đã bao gồm bias) dùng cho similar_items.
        _, item_embeddings = self._model.get_item_representations(features=item_features)
        self._item_embeddings = np.asarray(item_embeddings)

        log.info(
            "LightFM fitted: users=%d, items=%d, item_features=%s",
            interaction_matrix.shape[0],
            interaction_matrix.shape[1],
            None if item_features is None else item_features.shape,
        )

    # ---------------------------------------------------------- recommend
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Trả về top-n (item_idx, score) cho người dùng ``user_idx``."""
        if self._model is None or self._interaction is None:
            raise RuntimeError("LightFMRecommender chưa được fit.")

        n_items = self._interaction.shape[1]
        item_ids = np.arange(n_items, dtype=np.int32)

        scores: NDArray[Any] = self._model.predict(
            user_ids=user_idx,
            item_ids=item_ids,
            item_features=self._item_features,
            num_threads=self._num_threads,
        ).astype(np.float64)

        # Loại item đã xem bằng cách set score = -inf.
        if exclude_seen:
            seen = self._user_seen.get(user_idx, set())
            if seen:
                seen_idx = np.fromiter(seen, dtype=np.intp)
                scores[seen_idx] = -np.inf

        # Top-n hiệu quả bằng argpartition.
        if n < len(scores):
            top_indices = np.argpartition(-scores, n)[:n]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            top_indices = np.argsort(-scores)[:n]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ---------------------------------------------------- similar items
    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Trả về n item gần nhất theo cosine similarity trên embedding."""
        if self._item_embeddings is None:
            raise RuntimeError("LightFMRecommender chưa được fit.")

        target = self._item_embeddings[item_idx]
        target_norm = float(np.linalg.norm(target))
        if target_norm == 0.0:
            return []

        norms = np.linalg.norm(self._item_embeddings, axis=1)
        norms = np.where(norms == 0.0, 1.0, norms)

        sims: NDArray[Any] = (self._item_embeddings @ target) / (norms * target_norm)
        sims = sims.astype(np.float64, copy=True)
        # Loại chính nó ra khỏi kết quả.
        sims[item_idx] = -np.inf

        if n < len(sims):
            top_indices = np.argpartition(-sims, n)[:n]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
        else:
            top_indices = np.argsort(-sims)[:n]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    # ---------------------------------------------------------- accessors
    @property
    def model(self) -> LightFM:
        """Trả về model LightFM gốc (phục vụ MLflow hoặc joblib)."""
        if self._model is None:
            raise RuntimeError("LightFMRecommender chưa được fit.")
        return self._model

    @property
    def item_features(self) -> csr_matrix | None:
        """Ma trận item features đã dùng khi fit — serving cần để predict."""
        return self._item_features

    @property
    def item_embeddings(self) -> NDArray[Any]:
        """Embedding item kèm bias — dùng cho similar_items ở serving."""
        if self._item_embeddings is None:
            raise RuntimeError("LightFMRecommender chưa được fit.")
        return self._item_embeddings
