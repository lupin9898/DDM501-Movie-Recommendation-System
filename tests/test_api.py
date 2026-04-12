"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any  # used in _make_fake_artifact return type
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers — build a minimal in-memory model artifact so the app can start
# ---------------------------------------------------------------------------


def _make_fake_artifact(tmp_path: Path) -> dict[str, Any]:
    """Write a minimal model.pkl and supporting CSVs to tmp_path."""
    rng = np.random.default_rng(0)
    n_users, n_items, n_factors = 10, 20, 8

    artifact: dict[str, Any] = {
        "model_type": "als",
        "user_factors": rng.random((n_users, n_factors)).astype(np.float32),
        "item_factors": rng.random((n_items, n_factors)).astype(np.float32),
        "n_users": n_users,
        "n_items": n_items,
        "model_version": "als_test_v1",
        "hyperparams": {},
        "metrics": {},
    }

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(artifact, fh)

    # user mapping CSV
    import pandas as pd

    pd.DataFrame({"userId": range(n_users), "user_idx": range(n_users)}).to_csv(
        tmp_path / "user_mapping.csv", index=False
    )
    # item mapping CSV
    pd.DataFrame({"movieId": range(n_items), "movie_idx": range(n_items)}).to_csv(
        tmp_path / "item_mapping.csv", index=False
    )
    # movies CSV
    pd.DataFrame(
        {
            "movie_idx": range(n_items),
            "movieId": range(n_items),
            "title": [f"Movie {i}" for i in range(n_items)],
            "genres": ["Action|Drama"] * n_items,
        }
    ).to_csv(tmp_path / "movies.csv", index=False)

    # user_seen_items.pkl  (empty — no history)
    seen: dict[int, set[int]] = {i: set() for i in range(n_users)}
    with open(tmp_path / "user_seen_items.pkl", "wb") as fh:
        pickle.dump(seen, fh)

    return artifact


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a TestClient with a pre-loaded fake model."""
    _make_fake_artifact(tmp_path)

    # Patch settings so the app reads from tmp_path
    with patch("src.config.settings") as mock_settings:
        mock_settings.model_path = tmp_path / "model.pkl"
        mock_settings.data_processed_dir = tmp_path
        mock_settings.model_version = "als_test_v1"
        mock_settings.top_k = 10
        mock_settings.implicit_threshold = 3.5

        # Import after patching
        from src.serving.app import app
        from src.serving.recommender import RecommenderService

        # Build and load the service directly to bypass lifespan
        svc = RecommenderService()
        svc.load(model_path=tmp_path / "model.pkl", data_dir=tmp_path)
        app.state.recommender = svc  # type: ignore[attr-defined]

        with TestClient(app, raise_server_exceptions=True) as c:
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
