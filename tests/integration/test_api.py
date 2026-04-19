"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers — build a minimal in-memory model artifact so the app can start
# ---------------------------------------------------------------------------


def _make_fake_artifact(tmp_path: Path) -> dict[str, Any]:
    """Tạo artifact LightFM nhỏ để ``RecommenderService.load`` chạy đúng flow thật.

    Dùng LightFM thật với epochs=1, no_components=4 để pytest nhanh nhưng
    serving layer vẫn đi qua đường ``model.predict`` chuẩn.
    """
    from lightfm import LightFM

    rng = np.random.default_rng(0)
    n_users, n_items, n_genres, n_components = 10, 20, 3, 4

    # Ma trận tương tác giả lập — mỗi user tương tác với 3 item đầu của mình.
    rows = np.repeat(np.arange(n_users), 3)
    cols = np.concatenate([(u, (u + 1) % n_items, (u + 2) % n_items) for u in range(n_users)])
    data = np.ones(len(rows), dtype=np.float32)
    from scipy.sparse import coo_matrix

    interactions = coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()

    # Item features: mỗi phim gắn ngẫu nhiên 1 trong 3 genre.
    genre_rows = np.arange(n_items)
    genre_cols = rng.integers(0, n_genres, size=n_items)
    item_features = coo_matrix(
        (np.ones(n_items, dtype=np.float32), (genre_rows, genre_cols)),
        shape=(n_items, n_genres),
    ).tocsr()

    lightfm_model = LightFM(
        no_components=n_components,
        loss="warp",
        learning_rate=0.05,
        random_state=0,
    )
    lightfm_model.fit(interactions, item_features=item_features, epochs=1, num_threads=1)

    _, item_embeddings = lightfm_model.get_item_representations(features=item_features)

    # user_seen từ interactions.
    csr = interactions.tocsr()
    user_seen = {
        int(u): {int(i) for i in csr.indices[csr.indptr[u] : csr.indptr[u + 1]]}
        for u in range(n_users)
    }

    movies_df = pd.DataFrame(
        {
            "movieId": list(range(n_items)),
            "title": [f"Movie {i}" for i in range(n_items)],
            "genres": ["Action|Drama"] * n_items,
        }
    )

    artifact: dict[str, Any] = {
        "model_type": "lightfm",
        "model": lightfm_model,
        "item_features": item_features,
        "item_embeddings": np.asarray(item_embeddings),
        "user_id_map": {i: i for i in range(n_users)},
        "item_id_map": {i: i for i in range(n_items)},
        "reverse_item_id_map": {i: i for i in range(n_items)},
        "reverse_user_id_map": {i: i for i in range(n_users)},
        "user_seen": user_seen,
        "movies": movies_df,
        "n_users": n_users,
        "n_items": n_items,
        "model_version": "lightfm_test_v1",
        "hyperparams": {"no_components": n_components},
        "metrics": {},
        "num_threads": 1,
    }

    joblib.dump(artifact, tmp_path / "model.pkl")

    # Giữ ``movies.parquet`` phòng trường hợp RecommenderService fallback sang disk.
    movies_df.to_parquet(tmp_path / "movies.parquet", index=False)

    return artifact


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Return a TestClient with a pre-loaded fake model."""
    _make_fake_artifact(tmp_path)

    from src.config import settings
    from src.serving import app as app_module

    with (
        patch.object(settings, "model_path", tmp_path / "model.pkl"),
        patch.object(settings, "data_processed_dir", tmp_path),
        patch.object(settings, "model_version", "lightfm_test_v1"),
        patch.object(settings, "top_k", 10),
        patch.object(settings, "implicit_threshold", 3.5),
        TestClient(app_module.app, raise_server_exceptions=True) as c,
    ):
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_version" in data
        assert "uptime" in data


# ---------------------------------------------------------------------------
# /recommend endpoint
# ---------------------------------------------------------------------------


class TestRecommendEndpoint:
    def test_valid_user_returns_200(self, client: TestClient) -> None:
        resp = client.post("/recommend", json={"user_id": 0, "top_k": 5})
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        data = client.post("/recommend", json={"user_id": 0, "top_k": 5}).json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "model_version" in data
        assert "latency_ms" in data
        recs = data["recommendations"]
        assert len(recs) == 5
        for rec in recs:
            assert "movie_id" in rec
            assert "title" in rec
            assert "score" in rec
            assert "genres" in rec

    def test_recommendation_count_matches_top_k(self, client: TestClient) -> None:
        for k in (3, 5, 10):
            data = client.post("/recommend", json={"user_id": 0, "top_k": k}).json()
            assert len(data["recommendations"]) == k

    def test_unknown_user_returns_fallback(self, client: TestClient) -> None:
        """An unknown user_id should get popular-item fallback, not a 4xx."""
        resp = client.post("/recommend", json={"user_id": 9999, "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) > 0

    def test_latency_under_threshold(self, client: TestClient) -> None:
        start = time.perf_counter()
        client.post("/recommend", json={"user_id": 0, "top_k": 10})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f}ms exceeded 200ms"


# ---------------------------------------------------------------------------
# /recommend/batch endpoint
# ---------------------------------------------------------------------------


class TestBatchRecommendEndpoint:
    def test_batch_returns_200(self, client: TestClient) -> None:
        resp = client.post("/recommend/batch", json={"user_ids": [0, 1, 2], "top_k": 5})
        assert resp.status_code == 200

    def test_batch_result_count(self, client: TestClient) -> None:
        user_ids = [0, 1, 2]
        data = client.post("/recommend/batch", json={"user_ids": user_ids, "top_k": 5}).json()
        assert "results" in data
        assert len(data["results"]) == len(user_ids)

    def test_batch_latency(self, client: TestClient) -> None:
        start = time.perf_counter()
        client.post("/recommend/batch", json={"user_ids": list(range(10)), "top_k": 10})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500


# ---------------------------------------------------------------------------
# /similar/{movie_id} endpoint
# ---------------------------------------------------------------------------


class TestSimilarEndpoint:
    def test_similar_returns_200(self, client: TestClient) -> None:
        resp = client.get("/similar/0")
        assert resp.status_code == 200

    def test_similar_count(self, client: TestClient) -> None:
        data = client.get("/similar/0").json()
        # response uses "similar" key for item-item results
        key = "similar" if "similar" in data else "recommendations"
        assert len(data[key]) > 0

    def test_unknown_movie_returns_404(self, client: TestClient) -> None:
        resp = client.get("/similar/99999")
        assert resp.status_code == 404
