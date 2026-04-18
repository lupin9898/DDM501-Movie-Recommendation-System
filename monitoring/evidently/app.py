"""Evidently drift detection service for recommendation system."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest
from pythonjsonlogger import jsonlogger


def _configure_logging() -> None:
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "name": "logger", "asctime": "timestamp"},
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False


log = logging.getLogger(__name__)

DRIFT_REFRESH_INTERVAL_SECONDS = int(os.getenv("EVIDENTLY_REFRESH_INTERVAL_SECONDS", "300"))

scheduler = AsyncIOScheduler()


async def _refresh_drift_metrics_async() -> None:
    """Run the sync drift refresh on a worker thread so the event loop stays responsive."""
    _configure_logging()
    await asyncio.get_running_loop().run_in_executor(None, _refresh_drift_metrics)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    scheduler.add_job(
        _refresh_drift_metrics_async,
        "interval",
        seconds=DRIFT_REFRESH_INTERVAL_SECONDS,
        next_run_time=None,
        id="drift_refresh",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    log.info(
        "drift_refresh_scheduled",
        extra={
            "event": "drift_refresh_scheduled",
            "interval_seconds": DRIFT_REFRESH_INTERVAL_SECONDS,
        },
    )
    try:
        await _refresh_drift_metrics_async()
    except Exception as exc:  # noqa: BLE001
        log.warning("Initial drift refresh failed: %s", exc)
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="Evidently Drift Detection", version="1.0.0", lifespan=lifespan)

# ── Prometheus metrics ──────────────────────────────────────────────────────

DRIFT_SCORE = Gauge("evidently_drift_score", "Overall dataset drift score")
DRIFT_DETECTED = Gauge("evidently_drift_detected", "1 if drift detected, 0 otherwise")
DRIFT_REPORTS_TOTAL = Counter("evidently_reports_total", "Total drift reports generated")

RECSYS_DATA_DIR = Path(os.getenv("RECSYS_DATA_DIR", "/app/recsys-data"))
EVIDENTLY_DATA_DIR = Path("/app/data")
EVIDENTLY_DATA_DIR.mkdir(parents=True, exist_ok=True)

_last_report: dict | None = None


def _load_reference_data() -> pd.DataFrame | None:
    """Load reference dataset (training data statistics)."""
    ref_path = EVIDENTLY_DATA_DIR / "reference.parquet"
    if ref_path.exists():
        return pd.read_parquet(ref_path)
    # Fall back to processed train data
    train_path = RECSYS_DATA_DIR / "processed" / "train.parquet"
    if train_path.exists():
        df = pd.read_parquet(train_path)
        # Sample to keep reference lightweight
        ref = df.sample(min(10_000, len(df)), random_state=42)
        ref.to_parquet(ref_path, index=False)
        return ref
    return None


def _load_current_data() -> pd.DataFrame | None:
    """Load recent request log data for drift comparison."""
    current_path = EVIDENTLY_DATA_DIR / "current.parquet"
    if current_path.exists():
        return pd.read_parquet(current_path)
    # No current data yet — generate synthetic sample for demo
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_idx": rng.integers(0, 500, size=500),
            "movie_idx": rng.integers(0, 1000, size=500),
            "rating": rng.uniform(1.0, 5.0, size=500),
        }
    )


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4",
    )


def _compute_drift_report() -> tuple[Report, float, float]:
    """Build an Evidently drift report and update Prometheus metrics.

    Raises:
        HTTPException: If reference data is unavailable or datasets don't share columns.
    """
    global _last_report  # noqa: PLW0603

    reference = _load_reference_data()
    current = _load_current_data()

    if reference is None:
        raise HTTPException(
            status_code=503,
            detail="Reference data not available. Run preprocessing first.",
        )
    if current is None:
        raise HTTPException(status_code=503, detail="Current data not available.")

    cols = list(set(reference.columns) & set(current.columns))
    if not cols:
        raise HTTPException(status_code=500, detail="No common columns between datasets.")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])

    result = report.as_dict()
    drift_info = result.get("metrics", [{}])[0].get("result", {})
    drift_score = float(drift_info.get("dataset_drift_share", 0.0))
    drift_flag = 1.0 if drift_info.get("dataset_drift", False) else 0.0

    DRIFT_SCORE.set(drift_score)
    DRIFT_DETECTED.set(drift_flag)
    DRIFT_REPORTS_TOTAL.inc()

    _last_report = {"drift_score": drift_score, "drift_detected": bool(drift_flag)}
    log.info(
        "drift_report_generated",
        extra={
            "event": "drift_report_generated",
            "service": "evidently",
            "drift_score": drift_score,
            "drift_detected": bool(drift_flag),
            "reference_rows": int(len(reference)),
            "current_rows": int(len(current)),
            "common_columns": cols,
        },
    )
    return report, drift_score, drift_flag


def _refresh_drift_metrics() -> None:
    """Scheduled job: recompute drift and update Prometheus metrics."""
    try:
        _compute_drift_report()
    except HTTPException as exc:
        log.warning("Drift refresh skipped: %s", exc.detail)
    except Exception as exc:  # noqa: BLE001
        log.exception("Drift refresh failed: %s", exc)


@app.get("/report", response_class=HTMLResponse)
def drift_report() -> HTMLResponse:
    """Generate and return an HTML drift report."""
    report, _, _ = _compute_drift_report()
    report_path = EVIDENTLY_DATA_DIR / "last_report.html"
    report.save_html(str(report_path))
    return HTMLResponse(content=report_path.read_text())


@app.get("/status")
def status() -> dict:
    """Return latest drift status without regenerating report."""
    if _last_report is None:
        return {"status": "no_report", "message": "Call /report first"}
    return {"status": "ok", **_last_report}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8001)
