"""Prometheus metric definitions shared across routers.

Defined at module import time so the default Prometheus registry contains a
single canonical copy of each metric — re-importing (e.g. inside tests) must
not create duplicates.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

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
