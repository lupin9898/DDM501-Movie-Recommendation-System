"""ALS-based collaborative filtering recommender using the implicit library."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from src.models.baseline import BaseRecommender

log = logging.getLogger(__name__)


class ALSRecommender(BaseRecommender):
    """Alternating Least Squares recommender for implicit feedback."""

    def __init__(
        self,
        factors: int = 100,
        regularization: float = 0.05,
        iterations: int = 30,
        alpha: float = 40.0,
        use_bm25: bool = True,
        bm25_k1: float = 100.0,
        bm25_b: float = 0.8,
    ) -> None:
        self._factors = factors
        self._regularization = regularization
        self._iterations = iterations
        self._alpha = alpha
        self._use_bm25 = use_bm25
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

        self._model: AlternatingLeastSquares | None = None
        self._interaction: csr_matrix | None = None
        self._user_seen: dict[int, set[int]] = {}

    # -- fit / recommend -------------------------------------------------------

    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> None:
        """Train ALS on *interaction_matrix* (users x items).

        The matrix is multiplied by ``alpha`` to produce confidence weights
        before being passed to the ``implicit`` library.
        """
        self._interaction = interaction_matrix

        # Pre-compute seen items per user.
        self._user_seen = {}
        csr = interaction_matrix.tocsr()
        for uid in range(csr.shape[0]):
            start, end = int(csr.indptr[uid]), int(csr.indptr[uid + 1])
            self._user_seen[uid] = {int(x) for x in csr.indices[start:end]}

        # BM25 weighting downweights popular items so the model can learn
        # tail/niche item factors — breaks the popularity collapse.
        if self._use_bm25:
            weighted = bm25_weight(interaction_matrix, K1=self._bm25_k1, B=self._bm25_b).tocsr()
        else:
            weighted = interaction_matrix.tocsr()

        # Confidence-weighted matrix.
        confidence = (weighted * self._alpha).astype(np.float32)

        self._model = AlternatingLeastSquares(
            factors=self._factors,
            regularization=self._regularization,
            iterations=self._iterations,
        )
        # implicit.fit() expects a (users, items) CSR matrix
        self._model.fit(confidence.tocsr())

        log.info(
            "ALSRecommender fitted: factors=%d, iterations=%d, users=%d, items=%d, bm25=%s",
            self._factors,
            self._iterations,
            interaction_matrix.shape[0],
            interaction_matrix.shape[1],
            self._use_bm25,
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-*n* items for *user_idx* via dot-product scoring."""
        if self._model is None or self._interaction is None:
            raise RuntimeError("Model has not been fitted yet.")

        user_vec = self.user_factors[user_idx]
        scores: NDArray[Any] = user_vec @ self.item_factors.T

        if exclude_seen:
            seen = self._user_seen.get(user_idx, set())
            for idx in seen:
                scores[idx] = -np.inf

        if n < len(scores):
            top_indices = np.argpartition(-scores, n)[:n]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            top_indices = np.argsort(-scores)[:n]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # -- similar items ---------------------------------------------------------

    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Return *n* items most similar to *item_idx* (by latent factors)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        ids, scores = self._model.similar_items(item_idx, N=n + 1)
        # The first result is the item itself; skip it.
        results: list[tuple[int, float]] = []
        for item_id, score in zip(ids, scores, strict=True):
            if int(item_id) == item_idx:
                continue
            results.append((int(item_id), float(score)))
        return results[:n]

    # -- factor accessors ------------------------------------------------------

    @property
    def user_factors(self) -> NDArray[Any]:
        """User latent-factor matrix of shape (n_users, factors)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._model.user_factors)

    @property
    def item_factors(self) -> NDArray[Any]:
        """Item latent-factor matrix of shape (n_items, factors)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._model.item_factors)
