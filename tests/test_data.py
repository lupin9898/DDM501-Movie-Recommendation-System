"""Tests for data ingestion and preprocessing."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ratings() -> pd.DataFrame:
    """Create sample ratings data for testing."""
    rng = np.random.default_rng(42)
    n_ratings = 500
    n_users = 30
    n_movies = 50
    return pd.DataFrame(
        {
            "userId": rng.integers(1, n_users + 1, size=n_ratings),
            "movieId": rng.integers(1, n_movies + 1, size=n_ratings),
            "rating": rng.choice(
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], size=n_ratings
            ),
            "timestamp": sorted(rng.integers(946684800, 1704067200, size=n_ratings)),
        }
    )


@pytest.fixture
def sample_movies() -> pd.DataFrame:
    """Create sample movies data for testing."""
    genres = ["Action", "Comedy", "Drama", "Thriller", "Romance"]
    movies = []
    for i in range(1, 51):
        g = "|".join(np.random.default_rng(i).choice(genres, size=2, replace=False))
        movies.append({"movieId": i, "title": f"Movie {i} ({1990 + i % 30})", "genres": g})
    return pd.DataFrame(movies)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_filter_cold_start(self, sample_ratings: pd.DataFrame) -> None:
        """Users/items below threshold should be filtered."""
        from src.data.preprocessing import filter_cold_start

        filtered = filter_cold_start(
            sample_ratings, min_user_ratings=10, min_item_ratings=5
        )
        user_counts = filtered["userId"].value_counts()
        item_counts = filtered["movieId"].value_counts()
        assert user_counts.min() >= 10
        assert item_counts.min() >= 5

    def test_encode_ids(self, sample_ratings: pd.DataFrame) -> None:
        """Encoded IDs should be contiguous starting from 0."""
        from src.data.preprocessing import encode_ids

        encoded, user_map, movie_map = encode_ids(sample_ratings)
        assert encoded["user_idx"].min() == 0
        assert encoded["movie_idx"].min() == 0
        assert encoded["user_idx"].max() == encoded["user_idx"].nunique() - 1
        assert encoded["movie_idx"].max() == encoded["movie_idx"].nunique() - 1

    def test_encode_ids_no_original_cols(self, sample_ratings: pd.DataFrame) -> None:
        """Encoded DataFrame should not retain original userId/movieId columns."""
        from src.data.preprocessing import encode_ids

        encoded, _, _ = encode_ids(sample_ratings)
        assert "userId" not in encoded.columns
        assert "movieId" not in encoded.columns

    def test_temporal_split(self, sample_ratings: pd.DataFrame) -> None:
        """Train/val/test split should respect temporal ordering."""
        from src.data.preprocessing import temporal_split

        train, val, test = temporal_split(sample_ratings, train_frac=0.7, val_frac=0.15)
        assert len(train) + len(val) + len(test) == len(sample_ratings)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_temporal_split_size(self, sample_ratings: pd.DataFrame) -> None:
        """Split proportions should be approximately correct."""
        from src.data.preprocessing import temporal_split

        train, val, test = temporal_split(sample_ratings, train_frac=0.7, val_frac=0.15)
        total = len(sample_ratings)
        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02

    def test_temporal_ordering_train_test(self, sample_ratings: pd.DataFrame) -> None:
        """Max train timestamp must not exceed min test timestamp."""
        from src.data.preprocessing import temporal_split

        train, _val, test = temporal_split(sample_ratings, train_frac=0.7, val_frac=0.15)
        assert train["timestamp"].max() <= test["timestamp"].min()


class TestInteraction:
    """Tests for interaction matrix building."""

    def test_interaction_matrix_shape(self) -> None:
        """Interaction matrix should have correct dimensions."""
        from src.features.interaction import build_interaction_matrix

        ratings = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 1, 2],
                "movie_idx": [0, 1, 1, 2, 0],
                "rating": [4.0, 3.0, 5.0, 2.0, 4.5],
            }
        )
        explicit, implicit = build_interaction_matrix(
            ratings, n_users=3, n_items=3, implicit_threshold=3.5
        )
        assert explicit.shape == (3, 3)
        assert implicit.shape == (3, 3)

    def test_implicit_threshold(self) -> None:
        """Implicit matrix should binarize at threshold."""
        from src.features.interaction import build_interaction_matrix

        ratings = pd.DataFrame(
            {
                "user_idx": [0, 0, 1],
                "movie_idx": [0, 1, 0],
                "rating": [4.0, 2.0, 3.5],
            }
        )
        _, implicit = build_interaction_matrix(
            ratings, n_users=2, n_items=2, implicit_threshold=3.5
        )
        assert implicit[0, 0] == 1.0  # 4.0 >= 3.5
        assert implicit[0, 1] == 0.0  # 2.0 < 3.5
        assert implicit[1, 0] == 1.0  # 3.5 >= 3.5

    def test_user_seen_items(self) -> None:
        """User seen items should correctly map users to items."""
        from src.features.interaction import build_interaction_matrix, get_user_seen_items

        ratings = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 2],
                "movie_idx": [0, 1, 1, 2],
                "rating": [4.0, 3.0, 5.0, 4.5],
            }
        )
        explicit, _ = build_interaction_matrix(ratings, n_users=3, n_items=3)
        seen = get_user_seen_items(explicit)
        assert seen[0] == {0, 1}
        assert seen[1] == {1}
        assert seen[2] == {2}
