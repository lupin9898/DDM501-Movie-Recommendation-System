"""Baseline recommenders: random and popularity-based."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Common interface for all recommenders."""

    @abstractmethod
    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> None:
        """Train the model on a user-item interaction matrix."""

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-n recommendations as (item_idx, score) pairs."""


class RandomRecommender(BaseRecommender):
    """Recommend random unseen items -- useful as a lower-bound baseline."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._interaction: csr_matrix | None = None
        self._n_items: int = 0
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> None:
        """Store the interaction matrix and precompute seen items per user."""
        self._interaction = interaction_matrix
        self._n_items = interaction_matrix.shape[1]
        self._user_seen = {}
        for uid in range(interaction_matrix.shape[0]):
            self._user_seen[uid] = set(interaction_matrix[uid].indices.tolist())
        log.info(
            "RandomRecommender fitted: %d users, %d items",
            interaction_matrix.shape[0],
            self._n_items,
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return *n* random items not already seen by the user."""
        if self._interaction is None:
            raise RuntimeError("Model has not been fitted yet.")

        candidates = np.arange(self._n_items)
        if exclude_seen:
            seen = self._user_seen.get(user_idx, set())
            candidates = np.array([i for i in candidates if i not in seen])

        if len(candidates) == 0:
            return []

        chosen = self._rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        # Uniform score for random recommendations.
        return [(int(item), 1.0 / n) for item in chosen]


class PopularityRecommender(BaseRecommender):
    """Recommend the most globally popular items not yet seen by the user."""

    def __init__(self) -> None:
        self._interaction: csr_matrix | None = None
        self._item_popularity: NDArray[Any] = np.array([])
        self._popular_ranking: NDArray[Any] = np.array([])
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> None:
        """Compute global item popularity (number of interactions)."""
        self._interaction = interaction_matrix
        # Sum interactions per item (column-wise).
        self._item_popularity = np.asarray(interaction_matrix.sum(axis=0)).flatten()
        # Items sorted by descending popularity.
        self._popular_ranking = np.argsort(-self._item_popularity)

        self._user_seen = {}
        for uid in range(interaction_matrix.shape[0]):
            self._user_seen[uid] = set(interaction_matrix[uid].indices.tolist())

        log.info(
            "PopularityRecommender fitted: %d users, %d items",
            interaction_matrix.shape[0],
            interaction_matrix.shape[1],
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-*n* popular items the user has not interacted with."""
        if self._interaction is None:
            raise RuntimeError("Model has not been fitted yet.")

        seen = self._user_seen.get(user_idx, set()) if exclude_seen else set()
        results: list[tuple[int, float]] = []
        max_pop = float(self._item_popularity.max()) if self._item_popularity.max() > 0 else 1.0

        for item_idx in self._popular_ranking:
            if int(item_idx) in seen:
                continue
            score = float(self._item_popularity[item_idx]) / max_pop
            results.append((int(item_idx), score))
            if len(results) >= n:
                break

        return results
