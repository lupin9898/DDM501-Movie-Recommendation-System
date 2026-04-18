"""Centralized configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
    data_processed_dir: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"

    # Dataset
    dataset_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    dataset_name: str = "ml-25m"
    min_user_ratings: int = 20
    min_item_ratings: int = 10

    # Model
    model_path: Path = Path(__file__).resolve().parent.parent / "artifacts" / "model.pkl"
    model_version: str = "als_v1"
    top_k: int = 10
    implicit_threshold: float = 3.0

    # ALS defaults
    als_factors: int = 200
    als_regularization: float = 0.01
    als_iterations: int = 50
    als_alpha: float = 100.0

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "movielens-recommendation"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    model_config = {"env_prefix": "RECSYS_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
