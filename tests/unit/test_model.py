"""Tests for recommendation models and evaluation metrics."""

from __future__ import annotations

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
        user_seen = {0, 2, 3}
        assert rec_ids.isdisjoint(user_seen)

    def test_popularity_recommender_order(self, small_interaction_matrix: csr_matrix) -> None:
        """Popularity recommender should return items sorted by interaction count."""
        from src.models.baseline import PopularityRecommender

        model = PopularityRecommender()
        model.fit(small_interaction_matrix)
        recs = model.recommend(99, n=3, exclude_seen=False)
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


class TestLightFMRecommender:
    """Tests cho hybrid LightFM recommender."""

    @pytest.fixture
    def item_features(self) -> csr_matrix:
        """Ma trận 6×3 (3 genres) dạng sparse — mô phỏng genre one-hot."""
        dense = np.array(
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
        return csr_matrix(dense)

    @pytest.fixture
    def trained_lightfm(
        self,
        small_interaction_matrix: csr_matrix,
        item_features: csr_matrix,
    ) -> Any:
        """Fit tiny LightFM model — epochs=2, no_components=8 để CI nhanh."""
        from src.models.lightfm_hybrid import LightFMRecommender

        model = LightFMRecommender(
            no_components=8,
            loss="warp",
            learning_rate=0.05,
            epochs=2,
            num_threads=1,
            random_state=0,
        )
        model.fit(small_interaction_matrix, item_features=item_features)
        return model

    def test_lightfm_fits_without_error(
        self,
        small_interaction_matrix: csr_matrix,
        item_features: csr_matrix,
    ) -> None:
        from src.models.lightfm_hybrid import LightFMRecommender

        model = LightFMRecommender(no_components=8, epochs=2, num_threads=1, random_state=0)
        model.fit(small_interaction_matrix, item_features=item_features)

    def test_lightfm_recommendation_length(self, trained_lightfm: Any) -> None:
        recs = trained_lightfm.recommend(0, n=3, exclude_seen=False)
        assert len(recs) == 3

    def test_lightfm_excludes_seen(self, trained_lightfm: Any) -> None:
        recs = trained_lightfm.recommend(0, n=3, exclude_seen=True)
        rec_ids = {item_id for item_id, _ in recs}
        user0_seen = {0, 2, 3}
        assert rec_ids.isdisjoint(user0_seen)

    def test_lightfm_similar_items(self, trained_lightfm: Any) -> None:
        similar = trained_lightfm.similar_items(0, n=3)
        assert len(similar) == 3
        item_ids = {item_id for item_id, _ in similar}
        assert 0 not in item_ids

    def test_lightfm_item_embeddings_shape(self, trained_lightfm: Any) -> None:
        """Embedding item kèm bias phải khớp n_items."""
        assert trained_lightfm.item_embeddings.shape[0] == 6


class TestEvaluationMetrics:
    """Tests for ranking metrics."""

    def test_precision_at_k_perfect(self) -> None:
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_precision_at_k_zero(self) -> None:
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_precision_at_k_partial(self) -> None:
        from src.evaluation.metrics import precision_at_k

        assert precision_at_k([1, 4, 2], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_recall_at_k_perfect(self) -> None:
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_recall_at_k_partial(self) -> None:
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 4, 5], {1, 2}, k=3) == pytest.approx(0.5)

    def test_ndcg_at_k_perfect(self) -> None:
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_ndcg_at_k_zero(self) -> None:
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == pytest.approx(0.0)

    def test_f1_at_k_happy_path(self) -> None:
        from src.evaluation.metrics import f1_at_k

        assert f1_at_k(0.5, 0.5) == pytest.approx(0.5)
        assert f1_at_k(1.0, 0.5) == pytest.approx(2 / 3)

    def test_f1_at_k_zero_edge(self) -> None:
        from src.evaluation.metrics import f1_at_k

        assert f1_at_k(0.0, 0.0) == 0.0
        assert f1_at_k(0.0, 0.5) == 0.0

    def test_hit_rate_at_k_hit(self) -> None:
        from src.evaluation.metrics import hit_rate_at_k

        assert hit_rate_at_k([5, 1, 9], {1, 2}, k=3) == 1.0

    def test_hit_rate_at_k_miss(self) -> None:
        from src.evaluation.metrics import hit_rate_at_k

        assert hit_rate_at_k([5, 7, 9], {1, 2}, k=3) == 0.0

    def test_reciprocal_rank_first(self) -> None:
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([1, 2, 3], {1, 5}) == pytest.approx(1.0)

    def test_reciprocal_rank_later(self) -> None:
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([9, 1, 3], {1}) == pytest.approx(0.5)
        assert reciprocal_rank([9, 8, 1], {1}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_no_hit(self) -> None:
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([5, 6, 7], {1, 2}) == 0.0

    def test_metrics_in_range(self) -> None:
        from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k

        rng = np.random.default_rng(0)
        recommended = list(rng.integers(0, 20, size=10))
        relevant = set(rng.integers(0, 20, size=5).tolist())
        assert 0.0 <= precision_at_k(recommended, relevant, k=10) <= 1.0
        assert 0.0 <= recall_at_k(recommended, relevant, k=10) <= 1.0
        assert 0.0 <= ndcg_at_k(recommended, relevant, k=10) <= 1.0

    def test_coverage(self) -> None:
        from src.evaluation.metrics import coverage

        all_recs = [[0, 1], [2, 3], [0, 2]]
        assert coverage(all_recs, n_items=4) == pytest.approx(1.0)
        assert coverage(all_recs, n_items=8) == pytest.approx(0.5)


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

        features = np.ones((3, 3), dtype=np.float32)
        assert diversity([0, 1, 2], features) == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_items_have_max_diversity(self) -> None:
        from src.evaluation.metrics import diversity

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
