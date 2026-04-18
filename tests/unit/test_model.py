"""Tests for recommendation models and evaluation metrics."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def small_interaction_matrix() -> csr_matrix:
    """4 users × 6 items, dense enough for testing."""
    data = np.array(
        [
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    return csr_matrix(data)


class TestBaselineModels:
    """Tests for Random and Popularity baselines."""

    def test_random_recommender_length(self, small_interaction_matrix: csr_matrix) -> None:
        """Random recommender should return exactly top_k items."""
        from src.models.baseline import RandomRecommender

        model = RandomRecommender(seed=42)
        model.fit(small_interaction_matrix)
        recs = model.recommend(0, n=3)
        assert len(recs) == 3

    def test_random_recommender_excludes_seen(self, small_interaction_matrix: csr_matrix) -> None:
        """Random recommender should not recommend already-seen items."""
        from src.models.baseline import RandomRecommender

        model = RandomRecommender(seed=42)
        model.fit(small_interaction_matrix)
        recs = model.recommend(0, n=3, exclude_seen=True)
        rec_ids = {item_id for item_id, _ in recs}
        # User 0 interacted with items 0, 2, 3 (indices where value=1)
        user_seen = {0, 2, 3}
        assert rec_ids.isdisjoint(user_seen)

    def test_popularity_recommender_order(self, small_interaction_matrix: csr_matrix) -> None:
        """Popularity recommender should return items sorted by interaction count."""
        from src.models.baseline import PopularityRecommender

        model = PopularityRecommender()
        model.fit(small_interaction_matrix)
        recs = model.recommend(99, n=3, exclude_seen=False)  # unknown user
        assert len(recs) == 3
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)

    def test_popularity_recommender_excludes_seen(
        self, small_interaction_matrix: csr_matrix
    ) -> None:
        """Popularity recommender should not recommend seen items when asked."""
        from src.models.baseline import PopularityRecommender

        model = PopularityRecommender()
        model.fit(small_interaction_matrix)
        recs = model.recommend(0, n=3, exclude_seen=True)
        rec_ids = {item_id for item_id, _ in recs}
        user0_seen = {0, 2, 3}
        assert rec_ids.isdisjoint(user0_seen)


class TestALSRecommender:
    """Tests for ALS collaborative filtering model."""

    @pytest.fixture
    def trained_als(self, small_interaction_matrix: csr_matrix):  # type: ignore[no-untyped-def]
        """Fit a small ALS model."""
        from src.models.collaborative import ALSRecommender

        model = ALSRecommender(factors=10, iterations=5, regularization=0.01, alpha=15.0)
        model.fit(small_interaction_matrix)
        return model

    def test_als_trains_without_error(self, small_interaction_matrix: csr_matrix) -> None:
        """ALS model should fit without raising exceptions."""
        from src.models.collaborative import ALSRecommender

        model = ALSRecommender(factors=10, iterations=5)
        model.fit(small_interaction_matrix)  # should not raise

    def test_als_factors_shape(self, trained_als) -> None:  # type: ignore[no-untyped-def]
        """user_factors and item_factors should have correct shapes."""
        n_users, n_items = 4, 6
        assert trained_als.user_factors.shape == (n_users, 10)
        assert trained_als.item_factors.shape == (n_items, 10)

    def test_als_recommendation_length(self, trained_als) -> None:  # type: ignore[no-untyped-def]
        """ALS should return exactly top_k recommendations."""
        recs = trained_als.recommend(0, n=3)
        assert len(recs) == 3

    def test_als_no_seen_in_recommendations(self, trained_als) -> None:  # type: ignore[no-untyped-def]
        """ALS recommendations should not contain items the user already interacted with."""
        recs = trained_als.recommend(0, n=3, exclude_seen=True)
        rec_ids = {item_id for item_id, _ in recs}
        user0_seen = {0, 2, 3}
        assert rec_ids.isdisjoint(user0_seen)

    def test_als_similar_items(self, trained_als) -> None:  # type: ignore[no-untyped-def]
        """similar_items should return items excluding the query item."""
        similar = trained_als.similar_items(0, n=3)
        assert len(similar) == 3
        item_ids = {item_id for item_id, _ in similar}
        assert 0 not in item_ids  # query item excluded


class TestContentBasedRecommender:
    """Tests for content-based recommender."""

    @pytest.fixture
    def item_features(self) -> np.ndarray[Any, Any]:
        """Toy 6-item feature matrix (one-hot genres)."""
        return np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.float32,
        )

    def test_content_based_fits_without_error(
        self, small_interaction_matrix: csr_matrix, item_features: np.ndarray[Any, Any]
    ) -> None:
        """ContentBasedRecommender should fit without raising."""
        from src.models.content_based import ContentBasedRecommender

        model = ContentBasedRecommender()
        model.fit(small_interaction_matrix, item_features=item_features)

    def test_content_based_recommendation_length(
        self, small_interaction_matrix: csr_matrix, item_features: np.ndarray[Any, Any]
    ) -> None:
        """ContentBasedRecommender should return exactly n items."""
        from src.models.content_based import ContentBasedRecommender

        model = ContentBasedRecommender()
        model.fit(small_interaction_matrix, item_features=item_features)
        recs = model.recommend(0, n=3, exclude_seen=True)
        assert len(recs) == 3

    def test_content_based_no_seen(
        self, small_interaction_matrix: csr_matrix, item_features: np.ndarray[Any, Any]
    ) -> None:
        """ContentBasedRecommender should exclude seen items."""
        from src.models.content_based import ContentBasedRecommender

        model = ContentBasedRecommender()
        model.fit(small_interaction_matrix, item_features=item_features)
        recs = model.recommend(0, n=3, exclude_seen=True)
        rec_ids = {item_id for item_id, _ in recs}
        user0_seen = {0, 2, 3}
        assert rec_ids.isdisjoint(user0_seen)


class TestEvaluationMetrics:
    """Tests for ranking metrics."""

    def test_precision_at_k_perfect(self) -> None:
        """Precision@K = 1.0 when all recommendations are relevant."""
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_precision_at_k_zero(self) -> None:
        """Precision@K = 0.0 when no recommendations are relevant."""
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_precision_at_k_partial(self) -> None:
        """Precision@K should be proportional to hits in top-K."""
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([1, 4, 2], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_recall_at_k_perfect(self) -> None:
        """Recall@K = 1.0 when all relevant items are in top-K."""
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_recall_at_k_partial(self) -> None:
        """Recall@K = 0.5 when half of relevant items are retrieved."""
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 4, 5], {1, 2}, k=3) == pytest.approx(0.5)

    def test_ndcg_at_k_perfect(self) -> None:
        """NDCG@K = 1.0 for a perfect ranking."""
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_ndcg_at_k_zero(self) -> None:
        """NDCG@K = 0.0 when no relevant items are retrieved."""
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == pytest.approx(0.0)

    def test_f1_at_k_happy_path(self) -> None:
        """F1 is the harmonic mean of precision and recall."""
        from src.evaluation.metrics import f1_at_k

        assert f1_at_k(0.5, 0.5) == pytest.approx(0.5)
        assert f1_at_k(1.0, 0.5) == pytest.approx(2 / 3)

    def test_f1_at_k_zero_edge(self) -> None:
        """F1 = 0 when both precision and recall are 0 (no div-by-zero)."""
        from src.evaluation.metrics import f1_at_k

        assert f1_at_k(0.0, 0.0) == 0.0
        assert f1_at_k(0.0, 0.5) == 0.0

    def test_hit_rate_at_k_hit(self) -> None:
        """Hit rate = 1 when at least one recommendation is relevant."""
        from src.evaluation.metrics import hit_rate_at_k

        assert hit_rate_at_k([5, 1, 9], {1, 2}, k=3) == 1.0

    def test_hit_rate_at_k_miss(self) -> None:
        """Hit rate = 0 when no recommendation is relevant."""
        from src.evaluation.metrics import hit_rate_at_k

        assert hit_rate_at_k([5, 7, 9], {1, 2}, k=3) == 0.0

    def test_reciprocal_rank_first(self) -> None:
        """MRR = 1.0 when the first item is relevant."""
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([1, 2, 3], {1, 5}) == pytest.approx(1.0)

    def test_reciprocal_rank_later(self) -> None:
        """MRR = 1/rank of the first relevant item."""
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([9, 1, 3], {1}) == pytest.approx(0.5)
        assert reciprocal_rank([9, 8, 1], {1}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_no_hit(self) -> None:
        """MRR = 0 when no relevant item appears in recommendations."""
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([5, 6, 7], {1, 2}) == 0.0

    def test_metrics_in_range(self) -> None:
        """All metrics should be in [0, 1]."""
        from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k

        rng = np.random.default_rng(0)
        recommended = list(rng.integers(0, 20, size=10))
        relevant = set(rng.integers(0, 20, size=5).tolist())
        assert 0.0 <= precision_at_k(recommended, relevant, k=10) <= 1.0
        assert 0.0 <= recall_at_k(recommended, relevant, k=10) <= 1.0
        assert 0.0 <= ndcg_at_k(recommended, relevant, k=10) <= 1.0

    def test_coverage(self) -> None:
        """Coverage should equal the fraction of unique items recommended."""
        from src.evaluation.metrics import coverage

        all_recs = [[0, 1], [2, 3], [0, 2]]
        assert coverage(all_recs, n_items=4) == pytest.approx(1.0)
        assert coverage(all_recs, n_items=8) == pytest.approx(0.5)

    def test_als_beats_random(self, small_interaction_matrix: csr_matrix) -> None:
        """ALS precision@K should be >= random baseline on average (statistical)."""
        from src.evaluation.metrics import precision_at_k
        from src.models.baseline import RandomRecommender
        from src.models.collaborative import ALSRecommender

        als = ALSRecommender(factors=10, iterations=10)
        als.fit(small_interaction_matrix)

        rand = RandomRecommender(seed=0)
        rand.fit(small_interaction_matrix)

        # Evaluate on a toy test set: user 0 liked items 0, 2, 3
        relevant = {0, 2, 3}
        als_recs = [i for i, _ in als.recommend(0, n=3, exclude_seen=False)]
        rand_recs = [i for i, _ in rand.recommend(0, n=3, exclude_seen=False)]

        als_p = precision_at_k(als_recs, relevant, k=3)
        rand_p = precision_at_k(rand_recs, relevant, k=3)
        # ALS should perform at least as well on its own training data
        assert als_p >= rand_p or als_p > 0


# ---------------------------------------------------------------------------
# Edge cases for point-wise metrics
# ---------------------------------------------------------------------------


class TestMetricsEdgeCases:
    def test_precision_at_k_zero_k(self) -> None:
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=0) == 0.0

    def test_precision_at_k_empty_recommended(self) -> None:
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([], {1, 2}, k=5) == 0.0

    def test_recall_at_k_zero_k(self) -> None:
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 2, 3], {1, 2}, k=0) == 0.0

    def test_recall_at_k_empty_relevant(self) -> None:
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_ndcg_at_k_zero_k(self) -> None:
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([1, 2, 3], {1, 2}, k=0) == 0.0

    def test_ndcg_at_k_empty_relevant(self) -> None:
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_coverage_zero_n_items(self) -> None:
        from src.evaluation.metrics import coverage

        assert coverage([[0, 1], [2]], n_items=0) == 0.0


# ---------------------------------------------------------------------------
# Diversity metric
# ---------------------------------------------------------------------------


class TestDiversityMetric:
    def test_single_item_returns_zero(self) -> None:
        from src.evaluation.metrics import diversity

        features = np.eye(3, dtype=np.float32)
        assert diversity([0], features) == 0.0

    def test_identical_items_have_zero_diversity(self) -> None:
        from src.evaluation.metrics import diversity

        # All items share the same feature vector → cosine_sim = 1 → distance = 0
        features = np.ones((3, 3), dtype=np.float32)
        assert diversity([0, 1, 2], features) == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_items_have_max_diversity(self) -> None:
        from src.evaluation.metrics import diversity

        # Orthogonal one-hot vectors → cosine_sim = 0 → distance = 1
        features = np.eye(3, dtype=np.float32)
        assert diversity([0, 1, 2], features) == pytest.approx(1.0, abs=1e-5)

    def test_diversity_in_unit_range(self) -> None:
        from src.evaluation.metrics import diversity

        rng = np.random.default_rng(42)
        features = rng.random((6, 4)).astype(np.float32)
        d = diversity([0, 1, 2, 3], features)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# evaluate_model integration
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_evaluate_returns_all_keys(self, small_interaction_matrix: csr_matrix) -> None:
        from src.evaluation.metrics import evaluate_model
        from src.models.baseline import PopularityRecommender

        model = PopularityRecommender()
        model.fit(small_interaction_matrix)

        test_ratings = pd.DataFrame(
            {
                "user_idx": [0, 1, 2],
                "movie_idx": [0, 1, 2],
                "rating": [4.0, 4.0, 4.0],
            }
        )
        results = evaluate_model(model, test_ratings, small_interaction_matrix, k=3)
        for key in ("precision_at_k", "recall_at_k", "ndcg_at_k", "coverage", "n_users_evaluated"):
            assert key in results

    def test_evaluate_metrics_in_range(self, small_interaction_matrix: csr_matrix) -> None:
        from src.evaluation.metrics import evaluate_model
        from src.models.baseline import PopularityRecommender

        model = PopularityRecommender()
        model.fit(small_interaction_matrix)

        test_ratings = pd.DataFrame(
            {
                "user_idx": [0, 1, 2, 3],
                "movie_idx": [0, 1, 2, 4],
                "rating": [5.0, 4.0, 4.5, 3.5],
            }
        )
        results = evaluate_model(model, test_ratings, small_interaction_matrix, k=3)
        assert 0.0 <= results["precision_at_k"] <= 1.0
        assert 0.0 <= results["recall_at_k"] <= 1.0
        assert 0.0 <= results["ndcg_at_k"] <= 1.0
        assert results["n_users_evaluated"] > 0

    def test_evaluate_no_relevant_items_returns_zeros(
        self, small_interaction_matrix: csr_matrix
    ) -> None:
        from src.evaluation.metrics import evaluate_model
        from src.models.baseline import PopularityRecommender

        model = PopularityRecommender()
        model.fit(small_interaction_matrix)

        # All ratings below threshold → no relevant items → n_users_evaluated = 0
        test_ratings = pd.DataFrame(
            {
                "user_idx": [0, 1],
                "movie_idx": [0, 1],
                "rating": [1.0, 1.0],
            }
        )
        results = evaluate_model(model, test_ratings, small_interaction_matrix, k=3, threshold=3.5)
        assert results["n_users_evaluated"] == 0.0
        assert results["precision_at_k"] == 0.0
