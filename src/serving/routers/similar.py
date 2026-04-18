"""Similar-items endpoint — /similar/{movie_id}."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query, Request

from src.serving import state
from src.serving.metrics import LATENCY_HISTOGRAM, REQUEST_COUNTER
from src.serving.recommender import UnknownMovieError
from src.serving.schemas import MovieRecommendation, SimilarMoviesResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/similar/{movie_id}", response_model=SimilarMoviesResponse)
async def similar_movies(
    http_request: Request,
    movie_id: int,
    top_k: int = Query(default=10, ge=1, le=500),
) -> SimilarMoviesResponse:
    """Find movies similar to the given movie ID."""
    start = time.perf_counter()
    endpoint = "/similar"
    request_id = getattr(http_request.state, "request_id", None)

    try:
        results = state.service.similar_items(movie_id=movie_id, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(latency_ms / 1000)
        REQUEST_COUNTER.labels(status="success", endpoint=endpoint).inc()

        similar = [
            MovieRecommendation(
                movie_id=int(r["movie_id"]),  # type: ignore[arg-type]
                title=str(r["title"]),
                score=float(r["score"]),  # type: ignore[arg-type]
                genres=list(r["genres"]),  # type: ignore[arg-type]
            )
            for r in results
        ]
        log.info(
            "similar_movies_served",
            extra={
                "event": "similar_movies_served",
                "request_id": request_id,
                "endpoint": endpoint,
                "movie_id": movie_id,
                "top_k": top_k,
                "result_count": len(similar),
                "latency_ms": round(latency_ms, 3),
                "status": "success",
            },
        )
        return SimilarMoviesResponse(movie_id=movie_id, similar=similar)

    except UnknownMovieError:
        REQUEST_COUNTER.labels(status="not_found", endpoint=endpoint).inc()
        log.warning(
            "similar_movies_not_found",
            extra={
                "event": "similar_movies_not_found",
                "request_id": request_id,
                "endpoint": endpoint,
                "movie_id": movie_id,
                "status": "not_found",
            },
        )
        raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found") from None

    except Exception:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.exception(
            "similar_movies_failed",
            extra={
                "event": "similar_movies_failed",
                "request_id": request_id,
                "endpoint": endpoint,
                "movie_id": movie_id,
                "status": "error",
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
