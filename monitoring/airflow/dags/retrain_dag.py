"""Airflow DAG: weekly model retraining pipeline."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

DATA_RAW = Path(os.getenv("RECSYS_DATA_RAW_DIR", "/opt/airflow/data/raw"))
DATA_PROCESSED = Path(os.getenv("RECSYS_DATA_PROCESSED_DIR", "/opt/airflow/data/processed"))
ARTIFACTS = Path(os.getenv("RECSYS_ARTIFACTS_DIR", "/opt/airflow/artifacts"))

default_args = {
    "owner": "recsys",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ── Branch helpers ──────────────────────────────────────────────────────────


def check_raw_data(**ctx: dict) -> str:
    """Branch: download data if missing, else skip to processed check."""
    if not (DATA_RAW / "ratings.csv").exists():
        return "download_data"
    return "check_processed_data"


def check_processed_data(**ctx: dict) -> str:
    """Branch: preprocess if missing, else skip straight to training."""
    if not (DATA_PROCESSED / "train.parquet").exists():
        return "preprocess_data"
    return "train_model"


# ── API reload ──────────────────────────────────────────────────────────────


def reload_api(**ctx: dict) -> None:
    """Restart the API container so it picks up the new model.pkl.

    model.pkl is written to the bind-mounted host dir (${RECSYS_ARTIFACTS_DIR})
    which is shared between /opt/airflow/artifacts here and /app/artifacts in
    the API container — no docker cp needed, only a restart.
    """
    import docker  # installed via monitoring/airflow/requirements.txt

    client = docker.from_env()
    # Find by compose service label — works regardless of project/container name
    containers = client.containers.list(filters={"label": "com.docker.compose.service=api"})
    if not containers:
        print("API container not running — skipping reload")
        return
    api_container = containers[0]
    print(f"Restarting API container: {api_container.name}")
    api_container.restart(timeout=30)
    print("API restarted — new model will be loaded on startup")


# ── DAG definition ──────────────────────────────────────────────────────────


with DAG(
    dag_id="recsys_weekly_retrain",
    default_args=default_args,
    description="Weekly movie recommendation model retraining",
    schedule="0 2 * * 0",  # Every Sunday at 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recsys", "training"],
) as dag:
    check_raw = BranchPythonOperator(
        task_id="check_raw_data",
        python_callable=check_raw_data,
    )

    download_data = BashOperator(
        task_id="download_data",
        bash_command="cd /opt/airflow && python -m src.data.ingestion",
    )

    check_processed = BranchPythonOperator(
        task_id="check_processed_data",
        python_callable=check_processed_data,
        trigger_rule="none_failed_min_one_success",
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="cd /opt/airflow && python -m src.data.preprocessing",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow && python -m src.training.train --model lightfm",
        trigger_rule="none_failed_min_one_success",
    )

    reload_api_task = PythonOperator(
        task_id="reload_api",
        python_callable=reload_api,
    )

    check_raw >> download_data >> check_processed
    check_raw >> check_processed
    check_processed >> preprocess_data >> train_model
    check_processed >> train_model
    train_model >> reload_api_task
