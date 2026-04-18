"""Download, extract, and persist the MovieLens-25M dataset."""

from __future__ import annotations

import hashlib
import io
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

from src.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IngestionError(Exception):
    """Raised when any ingestion step fails."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

EXPECTED_FILES: list[str] = ["ratings.csv", "movies.csv", "tags.csv"]

SCHEMA: dict[str, list[str]] = {
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "movies.csv": ["movieId", "title", "genres"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
}

DTYPES: dict[str, dict[str, str]] = {
    "ratings.csv": {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64",
    },
    "movies.csv": {
        "movieId": "int32",
        "title": "str",
        "genres": "str",
    },
    "tags.csv": {
        "userId": "int32",
        "movieId": "int32",
        "tag": "str",
        "timestamp": "int64",
    },
}


@dataclass
class IngestionMetadata:
    """Metadata produced after a successful ingestion run."""

    download_date: str
    zip_checksum_sha256: str
    row_counts: dict[str, int]
    output_dir: Path


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def download_dataset(url: str) -> bytes:
    """Download the zip archive from *url* and return raw bytes."""
    log.info("Downloading dataset from %s ...", url)
    response = requests.get(url, timeout=600, stream=True)
    response.raise_for_status()

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        total += len(chunk)
        if total % (50 * 1024 * 1024) == 0:
            log.info("  downloaded %d MB so far ...", total // (1024 * 1024))

    data = b"".join(chunks)
    log.info("Download complete — %d MB", len(data) // (1024 * 1024))
    return data


def compute_sha256(data: bytes) -> str:
    """Return hex-encoded SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def extract_csv_from_zip(
    zip_bytes: bytes,
    dataset_name: str,
    filenames: list[str],
) -> dict[str, pd.DataFrame]:
    """Extract the requested CSVs from the in-memory zip archive.

    Returns a mapping ``filename -> DataFrame``.
    """
    frames: dict[str, pd.DataFrame] = {}
    with (
        zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        tmp_path = Path(tmp_dir)
        for fname in filenames:
            member = f"{dataset_name}/{fname}"
            if member not in zf.namelist():
                raise IngestionError(f"Expected member '{member}' not found in archive")
            log.info("Extracting %s ...", member)
            extracted = tmp_path / fname
            extracted.write_bytes(zf.read(member))
            df = pd.read_csv(extracted, dtype=cast(Any, DTYPES.get(fname)))
            _validate_columns(df, fname)
            frames[fname] = df
            log.info("  %s → %d rows, %d columns", fname, len(df), len(df.columns))
    return frames


def _validate_columns(df: pd.DataFrame, filename: str) -> None:
    """Raise ``IngestionError`` if expected columns are missing."""
    expected = set(SCHEMA[filename])
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise IngestionError(f"{filename}: missing columns {missing}")


def save_raw_csvs(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist each DataFrame to *output_dir* as a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, df in frames.items():
        dest = output_dir / fname
        df.to_csv(dest, index=False)
        log.info("Saved %s (%d rows)", dest, len(df))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def ingest(
    url: str | None = None,
    dataset_name: str | None = None,
    output_dir: Path | None = None,
) -> IngestionMetadata:
    """Run the full ingestion pipeline: download ➜ extract ➜ save.

    Parameters default to values from :pydata:`src.config.settings`.
    """
    url = url or settings.dataset_url
    dataset_name = dataset_name or settings.dataset_name
    output_dir = output_dir or settings.data_raw_dir

    zip_bytes = download_dataset(url)
    checksum = compute_sha256(zip_bytes)
    log.info("SHA-256 checksum: %s", checksum)

    frames = extract_csv_from_zip(zip_bytes, dataset_name, EXPECTED_FILES)
    save_raw_csvs(frames, output_dir)

    row_counts = {fname: len(df) for fname, df in frames.items()}
    download_date = datetime.now(tz=UTC).isoformat()

    metadata = IngestionMetadata(
        download_date=download_date,
        zip_checksum_sha256=checksum,
        row_counts=row_counts,
        output_dir=output_dir,
    )
    log.info("Ingestion complete — metadata: %s", metadata)
    return metadata


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point for ``python -m src.data.ingestion``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    ingest()


if __name__ == "__main__":
    main()
