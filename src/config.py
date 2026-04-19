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
    min_user_ratings: int = 20
    min_item_ratings: int = 10

    # Model
    model_path: Path = Path(__file__).resolve().parent.parent / "artifacts" / "model.pkl"
    model_version: str = "lightfm_v1"
    top_k: int = 10
    implicit_threshold: float = 3.0

    # LightFM hybrid defaults — theo yêu cầu người dùng.
    lightfm_no_components: int = 64
    lightfm_loss: str = "warp"
    lightfm_learning_rate: float = 0.05
    lightfm_epochs: int = 30
    lightfm_num_threads: int = 4
    lightfm_test_size: float = 0.2
    lightfm_split_seed: int = 42

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "movielens-recommendation"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    model_config = {"env_prefix": "RECSYS_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
