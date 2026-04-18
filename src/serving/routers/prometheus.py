"""Prometheus scrape endpoint — /metrics."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics in text-exposition format."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
