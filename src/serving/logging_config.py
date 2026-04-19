"""Structured JSON logging configuration for the FastAPI recommendation service.

Emits single-line JSON to stdout so Filebeat can ship to Elasticsearch and
Kibana can index every field for search/aggregation without re-parsing.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter

SERVICE_NAME = os.getenv("RECSYS_SERVICE_NAME", "recsys-api")
SERVICE_VERSION = os.getenv("RECSYS_MODEL_VERSION", "unknown")
ENVIRONMENT = os.getenv("RECSYS_ENV", "production")


class RecsysJsonFormatter(JsonFormatter):
    """JSON formatter that injects fixed service metadata on every record."""

    def add_fields(
        self,
        log_data: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_data, record, message_dict)

        log_data["timestamp"] = self.formatTime(record, self.datefmt)
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["service"] = SERVICE_NAME
        log_data["service_version"] = SERVICE_VERSION
        log_data["env"] = ENVIRONMENT

        if record.exc_info:
            log_data["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None

        log_data.pop("color_message", None)
        log_data.pop("taskName", None)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger and uvicorn loggers."""
    formatter = RecsysJsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "name": "logger"},
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(level.upper())
