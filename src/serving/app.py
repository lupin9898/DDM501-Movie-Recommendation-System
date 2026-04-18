"""FastAPI application for serving movie recommendations.

This module wires up the app: middleware, lifespan (loads the model), and
route registration. Endpoint logic lives in ``src/serving/routers/``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.config import settings
from src.serving import state
from src.serving.logging_config import configure_logging
from src.serving.routers import health, prometheus, recommend, similar

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load model once on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Load model artifacts on startup, clean up on shutdown."""
    state.reset_start_time()
    configure_logging(settings.log_level)

    try:
        log.info(
            "model_loading_started",
            extra={"event": "model_load", "model_path": str(settings.model_path)},
        )
        state.service.load(settings.model_path, settings.data_processed_dir)
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
# App + middleware
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
# Routes
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(similar.router)
app.include_router(prometheus.router)
