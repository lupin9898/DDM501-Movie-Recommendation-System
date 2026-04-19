"""Tests cho src.features.lightfm_dataset — build LightFMBundle từ CSV thô."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.features.lightfm_dataset import (
    LightFMBundle,
    _collect_all_genres,
    _parse_genres,
    build_lightfm_dataset,
)


class TestParseGenres:
    """Parse chuỗi genre pipe-delimited."""

    def test_parse_basic(self) -> None:
        assert _parse_genres("Action|Comedy|Drama") == ["Action", "Comedy", "Drama"]

    def test_parse_strips_whitespace(self) -> None:
        assert _parse_genres(" Action | Comedy ") == ["Action", "Comedy"]

    def test_parse_skips_no_genres_sentinel(self) -> None:
        # "(no genres listed)" là placeholder MovieLens — phải loại ra
        assert _parse_genres("(no genres listed)") == []

    def test_parse_skips_empty_segments(self) -> None:
        assert _parse_genres("Action||Comedy") == ["Action", "Comedy"]

    def test_parse_none(self) -> None:
        assert _parse_genres(None) == []

    def test_parse_single_genre(self) -> None:
        assert _parse_genres("Horror") == ["Horror"]


class TestCollectAllGenres:
    def test_collect_unique_sorted(self) -> None:
        movies = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["A", "B", "C"],
                "genres": ["Action|Comedy", "Comedy|Drama", "Action|Drama"],
            }
        )
        assert _collect_all_genres(movies) == ["Action", "Comedy", "Drama"]

    def test_collect_ignores_nan(self) -> None:
        movies = pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["A", "B"],
                "genres": ["Action", np.nan],
            }
        )
        assert _collect_all_genres(movies) == ["Action"]


@pytest.fixture
def tiny_csv_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Dựng cặp ratings.csv + movies.csv bé xíu để test build_lightfm_dataset."""
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "movieId": [10, 20, 10, 30, 20, 30, 10, 40, 20, 40],
            "rating": [4.0, 5.0, 3.5, 4.0, 2.5, 4.5, 5.0, 3.0, 4.0, 2.0],
            "timestamp": list(range(1_600_000_000, 1_600_000_010)),
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["Matrix", "Inception", "Toy Story", "Scary Movie"],
            "genres": [
                "Action|Sci-Fi",
                "Action|Sci-Fi|Thriller",
                "Animation|Children",
                "(no genres listed)",
            ],
        }
    )
    r_path = tmp_path / "ratings.csv"
    m_path = tmp_path / "movies.csv"
    ratings.to_csv(r_path, index=False)
    movies.to_csv(m_path, index=False)
    return r_path, m_path


class TestBuildLightFMDataset:
    def test_returns_bundle(self, tiny_csv_pair: tuple[Path, Path]) -> None:
        r, m = tiny_csv_pair
        bundle = build_lightfm_dataset(r, m, test_size=0.2, seed=42)
        assert isinstance(bundle, LightFMBundle)

    def test_shapes_match_universe(self, tiny_csv_pair: tuple[Path, Path]) -> None:
        r, m = tiny_csv_pair
        bundle = build_lightfm_dataset(r, m, test_size=0.2, seed=42)
        # 5 users × 4 items
        assert bundle.train_interactions.shape == (5, 4)
        assert bundle.test_interactions.shape == (5, 4)
        assert isinstance(bundle.train_interactions, csr_matrix)
        assert isinstance(bundle.test_interactions, csr_matrix)

    def test_item_features_include_all_genres(self, tiny_csv_pair: tuple[Path, Path]) -> None:
        r, m = tiny_csv_pair
        bundle = build_lightfm_dataset(r, m, test_size=0.2, seed=42)
        # Unique genres: Action, Animation, Children, Sci-Fi, Thriller => 5
        # LightFM thêm 1 cột identity mỗi item => ≥ n_items + n_genres
        assert bundle.item_features.shape[0] == 4  # 4 items
        assert bundle.all_genres == [
            "Action",
            "Animation",
            "Children",
            "Sci-Fi",
            "Thriller",
        ]

    def test_mappings_are_int_keyed(self, tiny_csv_pair: tuple[Path, Path]) -> None:
        r, m = tiny_csv_pair
        bundle = build_lightfm_dataset(r, m, test_size=0.2, seed=42)
        # Keys/values đều int để joblib pickle nhẹ + int-friendly API
        for k, v in bundle.user_id_map.items():
            assert isinstance(k, int) and isinstance(v, int)
        for k, v in bundle.item_id_map.items():
            assert isinstance(k, int) and isinstance(v, int)
        # Reverse maps đúng nghịch đảo
        assert bundle.reverse_item_id_map == {v: k for k, v in bundle.item_id_map.items()}
        assert bundle.reverse_user_id_map == {v: k for k, v in bundle.user_id_map.items()}

    def test_split_is_deterministic(self, tiny_csv_pair: tuple[Path, Path]) -> None:
        r, m = tiny_csv_pair
        b1 = build_lightfm_dataset(r, m, test_size=0.2, seed=42)
        b2 = build_lightfm_dataset(r, m, test_size=0.2, seed=42)
        # Cùng seed ⇒ cùng train matrix
        assert (b1.train_interactions.toarray() == b2.train_interactions.toarray()).all()
