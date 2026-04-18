"""FastAPI application for serving movie recommendations."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.config import settings
from src.serving.logging_config import configure_logging
from src.serving.recommender import ModelNotLoadedError, RecommenderService, UnknownMovieError
from src.serving.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    HealthResponse,
    MovieRecommendation,
    RecommendRequest,
    RecommendResponse,
    SimilarMoviesResponse,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNTER = Counter(
    "recommendation_requests_total",
    "Total recommendation requests",
    ["status", "endpoint"],
)

COLD_START_COUNTER = Counter(
    "cold_start_fallback_total",
    "Number of requests that fell back to popular items (cold start)",
)

LATENCY_HISTOGRAM = Histogram(
    "recommendation_latency_seconds",
    "Latency of recommendation requests in seconds",
    ["endpoint"],
)

# ---------------------------------------------------------------------------
# Service singleton
# ---------------------------------------------------------------------------

service = RecommenderService()
_start_time: float = time.monotonic()  # overwritten in lifespan, but safe from the start


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Load model artifacts on startup, clean up on shutdown."""
    global _start_time  # noqa: PLW0603
    _start_time = time.monotonic()

    configure_logging(settings.log_level)

    try:
        log.info(
            "model_loading_started",
            extra={"event": "model_load", "model_path": str(settings.model_path)},
        )
        service.load(settings.model_path, settings.data_processed_dir)
        log.info(
            "model_loaded",
            extra={
                "event": "model_loaded",
                "model_version": settings.model_version,
                "model_path": str(settings.model_path),
            },
        )
    except Exception:
        log.exception(
            "model_load_failed",
            extra={
                "event": "model_load_failed",
                "model_path": str(settings.model_path),
            },
        )

    yield

    log.info("service_shutdown", extra={"event": "service_shutdown"})


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Movie Recommendation API",
    description="Real-time movie recommendations powered by ALS collaborative filtering",
    version=settings.model_version,
    lifespan=lifespan,
)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request and emit structured access log."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            log.exception(
                "request_error",
                extra={
                    "event": "request_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else None,
                    "latency_ms": round(latency_ms, 3),
                },
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        log.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    """Format elapsed seconds into a human-readable uptime string."""
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _build_recommend_response(
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status and uptime."""
    uptime_seconds = time.monotonic() - _start_time
    return HealthResponse(
        status="healthy" if service.is_loaded else "unavailable",
        model_version=settings.model_version,
        uptime=_format_uptime(uptime_seconds),
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(http_request: Request, request: RecommendRequest) -> RecommendResponse:
    """Generate top-k recommendations for a single user."""
    start = time.perf_counter()
    endpoint = "/recommend"
    request_id = getattr(http_request.state, "request_id", None)

    try:
        is_cold_start = not service.is_known_user(request.user_id)
        results = service.recommend(
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

        return _build_recommend_response(request.user_id, results, latency_ms)

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


@app.post("/recommend/batch", response_model=BatchRecommendResponse)
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
            is_cold_start = not service.is_known_user(user_id)
            results = service.recommend(
                user_id=user_id,
                top_k=request.top_k,
                exclude_seen=True,
            )

            if is_cold_start:
                COLD_START_COUNTER.inc()
                cold_start_count += 1

            latency_ms = (time.perf_counter() - start) * 1000
            responses.append(_build_recommend_response(user_id, results, latency_ms))

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


@app.get("/similar/{movie_id}", response_model=SimilarMoviesResponse)
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
        results = service.similar_items(movie_id=movie_id, top_k=top_k)
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


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
