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
    """Write a minimal model and supporting Parquet artifacts to tmp_path.

    Schema mirrors what ``src.training.train`` produces so RecommenderService
    loads through the real code paths (not the identity-mapping fallback).
    """
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

    joblib.dump(artifact, tmp_path / "model.pkl")

    pd.DataFrame({"original_id": range(n_users), "encoded_id": range(n_users)}).to_parquet(
        tmp_path / "user_id_map.parquet", index=False
    )

    pd.DataFrame({"original_id": range(n_items), "encoded_id": range(n_items)}).to_parquet(
        tmp_path / "movie_id_map.parquet", index=False
    )

    pd.DataFrame(
        {
            "movieId": range(n_items),
            "title": [f"Movie {i}" for i in range(n_items)],
            "genres": ["Action|Drama"] * n_items,
        }
    ).to_parquet(tmp_path / "movies.parquet", index=False)

    # Train log so RecommenderService can build user_seen_items from it.
    pd.DataFrame(
        {
            "user_idx": list(range(n_users)),
            "movie_idx": list(range(n_users)),
        }
    ).to_parquet(tmp_path / "train.parquet", index=False)

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
        patch.object(settings, "model_version", "als_test_v1"),
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
