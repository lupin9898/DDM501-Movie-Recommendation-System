"""Evaluation metrics for recommender systems."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baseline import BaseRecommender

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-wise ranking metrics
# ---------------------------------------------------------------------------


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Proportion of recommended items in top-K that are relevant."""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Proportion of relevant items found in top-K."""
    if k <= 0 or len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if k <= 0 or len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]

    # DCG: sum of 1/log2(rank+1) for each relevant item in top_k
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG: all relevant items appear first
    ideal_hits = min(len(relevant), k)
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / np.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregate / diversity metrics
# ---------------------------------------------------------------------------


def coverage(all_recommendations: list[list[int]], n_items: int) -> float:
    """Fraction of items appearing at least once across all recommendation lists."""
    if n_items <= 0:
        return 0.0
    unique_items: set[int] = set()
    for rec_list in all_recommendations:
        unique_items.update(rec_list)
    return len(unique_items) / n_items


def diversity(recommendations: list[int], item_features: NDArray[Any]) -> float:
    """Average intra-list distance (1 - cosine_sim) between all item pairs.

    Parameters
    ----------
    recommendations:
        List of item indices in a single recommendation list.
    item_features:
        Item-feature matrix of shape (n_items, n_features).

    Returns
    -------
    float
        Mean pairwise distance in [0, 1].  Higher means more diverse.
    """
    if len(recommendations) < 2:
        return 0.0

    vecs = item_features[recommendations]
    sim_matrix: NDArray[Any] = cosine_similarity(vecs)

    # Extract upper-triangle (excluding diagonal) with numpy — O(n) vs O(n²) loop
    upper = sim_matrix[np.triu_indices(len(recommendations), k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(1.0 - upper))


# ---------------------------------------------------------------------------
# Full model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: BaseRecommender,
    test_ratings: pd.DataFrame,
    train_interaction: csr_matrix,
    k: int = 10,
    threshold: float = 3.5,
) -> dict[str, float]:
    """Evaluate a recommender model on a held-out test set.

    Parameters
    ----------
    model:
        A fitted recommender that implements ``BaseRecommender``.
    test_ratings:
        DataFrame with at least columns ``user_idx``, ``movie_idx``, ``rating``.
    train_interaction:
        The user-item interaction matrix the model was trained on (used for
        excluding already-seen items during recommendation).
    k:
        Number of recommendations to generate per user.
    threshold:
        Minimum rating in *test_ratings* for an item to count as relevant.

    Returns
    -------
    dict[str, float]
        Averaged precision@k, recall@k, ndcg@k, and catalog coverage.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    all_recs: list[list[int]] = []

    n_items = train_interaction.shape[1]

    # Group test ratings by user.
    user_groups = test_ratings.groupby("user_idx")

    for raw_user_idx, group in user_groups:
        uid = int(raw_user_idx)  # type: ignore[arg-type]

        # Relevant items: those rated >= threshold in the test set.
        relevant: set[int] = set(
            group.loc[group["rating"] >= threshold, "movie_idx"].astype(int).tolist()
        )
        if len(relevant) == 0:
            continue

        # Get model recommendations.
        try:
            recs = model.recommend(uid, n=k, exclude_seen=True)
        except (IndexError, KeyError):
            # User may not exist in the training matrix.
            continue

        rec_items = [item_idx for item_idx, _score in recs]

        precisions.append(precision_at_k(rec_items, relevant, k))
        recalls.append(recall_at_k(rec_items, relevant, k))
        ndcgs.append(ndcg_at_k(rec_items, relevant, k))
        all_recs.append(rec_items)

    n_evaluated = len(precisions)
    if n_evaluated == 0:
        log.warning("No users evaluated -- check test_ratings and threshold.")
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "coverage": 0.0,
            "n_users_evaluated": 0.0,
        }

    results: dict[str, float] = {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "coverage": coverage(all_recs, n_items),
        "n_users_evaluated": float(n_evaluated),
    }

    log.info(
        "Evaluation (k=%d): precision=%.4f  recall=%.4f  ndcg=%.4f  coverage=%.4f  users=%d",
        k,
        results["precision_at_k"],
        results["recall_at_k"],
        results["ndcg_at_k"],
        results["coverage"],
        n_evaluated,
    )

    return results
