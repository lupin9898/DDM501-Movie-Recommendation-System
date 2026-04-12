"""ALS-based collaborative filtering recommender using the implicit library."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from implicit.als import AlternatingLeastSquares
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
    ) -> None:
        self._factors = factors
        self._regularization = regularization
        self._iterations = iterations
        self._alpha = alpha

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
        for uid in range(interaction_matrix.shape[0]):
            self._user_seen[uid] = set(interaction_matrix[uid].indices.tolist())

        # Confidence-weighted matrix.
        confidence = (interaction_matrix * self._alpha).astype(np.float32)

        self._model = AlternatingLeastSquares(
            factors=self._factors,
            regularization=self._regularization,
            iterations=self._iterations,
        )
        # implicit.fit() expects a (users, items) CSR matrix
        self._model.fit(confidence.tocsr())

        log.info(
            "ALSRecommender fitted: factors=%d, iterations=%d, users=%d, items=%d",
            self._factors,
            self._iterations,
            interaction_matrix.shape[0],
            interaction_matrix.shape[1],
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
        scores: np.ndarray = user_vec @ self.item_factors.T

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
    def user_factors(self) -> np.ndarray:
        """User latent-factor matrix of shape (n_users, factors)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._model.user_factors)

    @property
    def item_factors(self) -> np.ndarray:
        """Item latent-factor matrix of shape (n_items, factors)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._model.item_factors)
