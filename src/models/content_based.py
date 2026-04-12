"""Content-based recommender using item feature similarity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baseline import BaseRecommender

log = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Recommend items whose features are similar to the user's profile.

    The user profile is a weighted average of item-feature vectors for items
    the user has interacted with.  Scores are cosine similarities between the
    user profile and every candidate item.

    This recommender serves as a fallback for cold-start users who have very
    few collaborative signals.
    """

    def __init__(self) -> None:
        self._interaction: csr_matrix | None = None
        self._item_features: np.ndarray | None = None
        self._user_profiles: np.ndarray | None = None
        self._user_seen: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        interaction_matrix: csr_matrix,
        item_features: np.ndarray | csr_matrix | None = None,
        **kwargs: Any,
    ) -> None:
        """Build user profiles from *interaction_matrix* and *item_features*.

        Parameters
        ----------
        interaction_matrix:
            User-item matrix (users x items).  Non-zero entries indicate
            interactions (implicit) or ratings (explicit).
        item_features:
            Item-feature matrix (items x features).  Dense or sparse.
        """
        if item_features is None:
            raise ValueError("item_features must be provided for ContentBasedRecommender.")

        self._interaction = interaction_matrix

        # Ensure item features are dense for downstream cosine_similarity.
        if issparse(item_features):
            self._item_features = np.asarray(item_features.toarray())  # type: ignore[union-attr]
        else:
            self._item_features = np.asarray(item_features)

        # Precompute seen items.
        self._user_seen = {}
        for uid in range(interaction_matrix.shape[0]):
            self._user_seen[uid] = set(interaction_matrix[uid].indices.tolist())

        # Build user profiles: weighted average of item features.
        n_users = interaction_matrix.shape[0]
        n_features = self._item_features.shape[1]
        self._user_profiles = np.zeros((n_users, n_features), dtype=np.float64)

        for uid in range(n_users):
            row = interaction_matrix[uid]
            item_indices = row.indices
            if len(item_indices) == 0:
                continue
            weights = np.asarray(row.data, dtype=np.float64)
            feature_vecs = self._item_features[item_indices]  # (k, features)
            weighted_sum = weights @ feature_vecs  # (features,)
            total_weight = weights.sum()
            if total_weight > 0:
                self._user_profiles[uid] = weighted_sum / total_weight

        log.info(
            "ContentBasedRecommender fitted: %d users, %d items, %d features",
            n_users,
            interaction_matrix.shape[1],
            n_features,
        )

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-*n* items by cosine similarity to the user profile."""
        if self._user_profiles is None or self._item_features is None:
            raise RuntimeError("Model has not been fitted yet.")

        user_profile = self._user_profiles[user_idx].reshape(1, -1)

        # If the user has a zero profile, return empty (cold-cold start).
        if np.allclose(user_profile, 0):
            return []

        scores: np.ndarray = cosine_similarity(user_profile, self._item_features).flatten()

        if exclude_seen:
            seen = self._user_seen.get(user_idx, set())
            for idx in seen:
                scores[idx] = -np.inf

        if n < len(scores):
            top_indices = np.argpartition(-scores, n)[:n]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            top_indices = np.argsort(-scores)[:n]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -np.inf]
