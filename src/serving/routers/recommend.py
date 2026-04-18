"""Recommendation endpoints — /recommend and /recommend/batch."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.config import settings
from src.serving import state
from src.serving.metrics import COLD_START_COUNTER, LATENCY_HISTOGRAM, REQUEST_COUNTER
from src.serving.recommender import ModelNotLoadedError
from src.serving.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    MovieRecommendation,
    RecommendRequest,
    RecommendResponse,
)

log = logging.getLogger(__name__)
router = APIRouter()


def _build_response(
    user_id: int,
    results: list[dict[str, int | str | float | list[str]]],
    latency_ms: float,
) -> RecommendResponse:
    """Convert raw recommendation dicts into a typed response."""
    recommendations = [
        MovieRecommendation(
            movie_id=int(r["movie_id"]),  # type: ignore[arg-type]
            title=str(r["title"]),
            score=float(r["score"]),  # type: ignore[arg-type]
            genres=list(r["genres"]),  # type: ignore[arg-type]
        )
        for r in results
    ]
    return RecommendResponse(
        user_id=user_id,
        recommendations=recommendations,
        model_version=settings.model_version,
        latency_ms=round(latency_ms, 3),
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(http_request: Request, request: RecommendRequest) -> RecommendResponse:
    """Generate top-k recommendations for a single user."""
    start = time.perf_counter()
    endpoint = "/recommend"
    request_id = getattr(http_request.state, "request_id", None)

    try:
        is_cold_start = not state.service.is_known_user(request.user_id)
        results = state.service.recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )

        if is_cold_start:
            COLD_START_COUNTER.inc()

        latency_ms = (time.perf_counter() - start) * 1000
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(latency_ms / 1000)
        REQUEST_COUNTER.labels(status="success", endpoint=endpoint).inc()

        log.info(
            "recommendation_served",
            extra={
                "event": "recommendation_served",
                "request_id": request_id,
                "endpoint": endpoint,
                "user_id": request.user_id,
                "top_k": request.top_k,
                "exclude_seen": request.exclude_seen,
                "cold_start": is_cold_start,
                "result_count": len(results),
                "latency_ms": round(latency_ms, 3),
                "model_version": settings.model_version,
                "status": "success",
            },
        )
        return _build_response(request.user_id, results, latency_ms)

    except ModelNotLoadedError:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.warning(
            "model_not_loaded",
            extra={
                "event": "model_not_loaded",
                "request_id": request_id,
                "endpoint": endpoint,
                "user_id": request.user_id,
                "status": "error",
            },
        )
        raise HTTPException(
            status_code=503, detail="Model not loaded — service is in degraded mode"
        ) from None

    except Exception:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.exception(
            "recommendation_failed",
            extra={
                "event": "recommendation_failed",
                "request_id": request_id,
                "endpoint": endpoint,
                "user_id": request.user_id,
                "status": "error",
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/recommend/batch", response_model=BatchRecommendResponse)
async def recommend_batch(
    http_request: Request, request: BatchRecommendRequest
) -> BatchRecommendResponse:
    """Generate recommendations for multiple users in a single request."""
    batch_start = time.perf_counter()
    endpoint = "/recommend/batch"
    request_id = getattr(http_request.state, "request_id", None)
    cold_start_count = 0

    try:
        responses: list[RecommendResponse] = []
        for user_id in request.user_ids:
            start = time.perf_counter()
            is_cold_start = not state.service.is_known_user(user_id)
            results = state.service.recommend(
                user_id=user_id,
                top_k=request.top_k,
                exclude_seen=True,
            )
            if is_cold_start:
                COLD_START_COUNTER.inc()
                cold_start_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            responses.append(_build_response(user_id, results, latency_ms))

        total_latency_ms = (time.perf_counter() - batch_start) * 1000
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(total_latency_ms / 1000)
        REQUEST_COUNTER.labels(status="success", endpoint=endpoint).inc()

        log.info(
            "batch_recommendation_served",
            extra={
                "event": "batch_recommendation_served",
                "request_id": request_id,
                "endpoint": endpoint,
                "user_count": len(request.user_ids),
                "cold_start_count": cold_start_count,
                "top_k": request.top_k,
                "latency_ms": round(total_latency_ms, 3),
                "model_version": settings.model_version,
                "status": "success",
            },
        )
        return BatchRecommendResponse(
            results=responses,
            latency_ms=round(total_latency_ms, 3),
        )

    except ModelNotLoadedError:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.warning(
            "model_not_loaded",
            extra={
                "event": "model_not_loaded",
                "request_id": request_id,
                "endpoint": endpoint,
                "status": "error",
            },
        )
        raise HTTPException(
            status_code=503, detail="Model not loaded — service is in degraded mode"
        ) from None

    except Exception:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.exception(
            "batch_recommendation_failed",
            extra={
                "event": "batch_recommendation_failed",
                "request_id": request_id,
                "endpoint": endpoint,
                "user_count": len(request.user_ids),
                "status": "error",
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
