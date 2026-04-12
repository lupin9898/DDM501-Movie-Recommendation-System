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


def check_raw_data(**ctx: dict) -> str:
    """Branch: download data if missing, else skip."""
    if not (DATA_RAW / "ratings.csv").exists():
        return "download_data"
    return "check_processed_data"


def check_processed_data(**ctx: dict) -> str:
    """Branch: preprocess if missing, else skip."""
    if not (DATA_PROCESSED / "train.parquet").exists():
        return "preprocess_data"
    return "train_model"


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
        bash_command="cd /opt/airflow && python -m src.training.train --model als",
        trigger_rule="none_failed_min_one_success",
    )

    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="""
            CONTAINER=$(docker ps --filter "name=api" --format "{{.ID}}" | head -1)
            if [ -n "$CONTAINER" ]; then
                docker cp {{ var.value.get('artifacts_dir', '/opt/airflow/artifacts') }}/model.pkl $CONTAINER:/app/artifacts/model.pkl
                docker restart $CONTAINER
                echo "API reloaded."
            else
                echo "API not running — skipping reload."
            fi
        """,
    )

    check_raw >> download_data >> check_processed
    check_raw >> check_processed
    check_processed >> preprocess_data >> train_model
    check_processed >> train_model
    train_model >> reload_api
