"""Process-wide singletons for the FastAPI serving layer.

Routers import from this module so there is a single RecommenderService
instance per process, loaded once at startup by ``app.py``'s lifespan.
"""

from __future__ import annotations

import time

from src.serving.recommender import RecommenderService

service: RecommenderService = RecommenderService()
start_time: float = time.monotonic()


def reset_start_time() -> None:
    """Reset the uptime counter — called at app startup."""
    global start_time  # noqa: PLW0603
    start_time = time.monotonic()
