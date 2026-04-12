"""FastAPI application for serving movie recommendations."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)

from src.config import settings
from src.serving.recommender import RecommenderService, UnknownMovieError
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
_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts on startup, clean up on shutdown."""
    global _start_time  # noqa: PLW0603
    _start_time = time.monotonic()

    log.info("Loading model artifacts...")
    service.load(settings.model_path, settings.data_processed_dir)
    log.info("Model loaded — service is ready")

    yield

    log.info("Shutting down recommendation service")


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
    """Attach a correlation ID to every request for end-to-end tracing."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        log.info(
            "Incoming %s %s [rid=%s]",
            request.method,
            request.url.path,
            request_id,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
            movie_id=int(r["movie_id"]),
            title=str(r["title"]),
            score=float(r["score"]),
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
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Generate top-k recommendations for a single user."""
    start = time.perf_counter()
    endpoint = "/recommend"

    try:
        results = service.recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )

        # Detect cold-start fallback: user not in mapping
        if request.user_id not in service._user_to_idx:
            COLD_START_COUNTER.inc()

        latency_ms = (time.perf_counter() - start) * 1000
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(latency_ms / 1000)
        REQUEST_COUNTER.labels(status="success", endpoint=endpoint).inc()

        return _build_recommend_response(request.user_id, results, latency_ms)

    except Exception:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.exception("Error generating recommendations for user_id=%d", request.user_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.post("/recommend/batch", response_model=BatchRecommendResponse)
async def recommend_batch(request: BatchRecommendRequest) -> BatchRecommendResponse:
    """Generate recommendations for multiple users in a single request."""
    batch_start = time.perf_counter()
    endpoint = "/recommend/batch"

    try:
        responses: list[RecommendResponse] = []
        for user_id in request.user_ids:
            start = time.perf_counter()
            results = service.recommend(
                user_id=user_id,
                top_k=request.top_k,
                exclude_seen=True,
            )

            if user_id not in service._user_to_idx:
                COLD_START_COUNTER.inc()

            latency_ms = (time.perf_counter() - start) * 1000
            responses.append(_build_recommend_response(user_id, results, latency_ms))

        total_latency_ms = (time.perf_counter() - batch_start) * 1000
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(total_latency_ms / 1000)
        REQUEST_COUNTER.labels(status="success", endpoint=endpoint).inc()

        return BatchRecommendResponse(
            results=responses,
            latency_ms=round(total_latency_ms, 3),
        )

    except Exception:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.exception("Error generating batch recommendations")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.get("/similar/{movie_id}", response_model=SimilarMoviesResponse)
async def similar_movies(movie_id: int, top_k: int = 10) -> SimilarMoviesResponse:
    """Find movies similar to the given movie ID."""
    start = time.perf_counter()
    endpoint = "/similar"

    try:
        results = service.similar_items(movie_id=movie_id, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(latency_ms / 1000)
        REQUEST_COUNTER.labels(status="success", endpoint=endpoint).inc()

        similar = [
            MovieRecommendation(
                movie_id=int(r["movie_id"]),
                title=str(r["title"]),
                score=float(r["score"]),
                genres=list(r["genres"]),  # type: ignore[arg-type]
            )
            for r in results
        ]
        return SimilarMoviesResponse(movie_id=movie_id, similar=similar)

    except UnknownMovieError:
        REQUEST_COUNTER.labels(status="not_found", endpoint=endpoint).inc()
        raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found") from None

    except Exception:
        REQUEST_COUNTER.labels(status="error", endpoint=endpoint).inc()
        log.exception("Error finding similar movies for movie_id=%d", movie_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
