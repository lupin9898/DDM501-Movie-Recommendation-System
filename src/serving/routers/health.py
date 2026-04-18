"""Health endpoint — reports service status, uptime, and loaded-model metadata."""

from __future__ import annotations

import time

from fastapi import APIRouter

from src.config import settings
from src.serving import state
from src.serving.schemas import HealthResponse

router = APIRouter()


def _format_uptime(seconds: float) -> str:
    """Format elapsed seconds into a compact human-readable uptime string."""
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


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status, uptime, and currently-loaded model metadata."""
    uptime_seconds = time.monotonic() - state.start_time
    meta = state.service.model_meta if state.service.is_loaded else {}
    return HealthResponse(
        status="healthy" if state.service.is_loaded else "unavailable",
        model_version=str(meta.get("model_version") or settings.model_version),
        uptime=_format_uptime(uptime_seconds),
        model_type=meta.get("model_type"),
        trained_at=meta.get("trained_at"),
        mlflow_run_id=meta.get("mlflow_run_id"),
        git_sha=meta.get("git_sha"),
    )
