"""Training pipeline with MLflow experiment tracking."""

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config import settings
from src.evaluation.metrics import coverage, evaluate_model
from src.features.interaction import build_interaction_matrix
from src.models.baseline import PopularityRecommender, RandomRecommender
from src.models.collaborative import ALSRecommender
from src.models.content_based import ContentBasedRecommender

log = logging.getLogger(__name__)


def load_processed_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load preprocessed parquet files."""
    data: dict[str, pd.DataFrame] = {
        "train": pd.read_parquet(data_dir / "train.parquet"),
        "val": pd.read_parquet(data_dir / "val.parquet"),
        "test": pd.read_parquet(data_dir / "test.parquet"),
        "movies": pd.read_parquet(data_dir / "movies.parquet"),
        "user_id_map": pd.read_parquet(data_dir / "user_id_map.parquet"),
        "movie_id_map": pd.read_parquet(data_dir / "movie_id_map.parquet"),
    }
    log.info(
        "Loaded data: train=%d, val=%d, test=%d, movies=%d",
        len(data["train"]),
        len(data["val"]),
        len(data["test"]),
        len(data["movies"]),
    )
    return data


def train_and_evaluate(
    model_type: str = "als",
    hyperparams: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train a model, evaluate it, and return (model, metrics)."""
    if hyperparams is None:
        hyperparams = {}

    data_dir = settings.data_processed_dir
    data = load_processed_data(data_dir)

    train_df = data["train"]
    val_df = data["val"]
    movies_df = data["movies"]

    # Use global ID maps so n_users/n_items covers the full encoded space,
    # not just users/items present in the train split.
    n_users = len(data["user_id_map"])
    n_items = len(data["movie_id_map"])

    log.info("Building interaction matrix: %d users, %d items", n_users, n_items)
    explicit_matrix, implicit_matrix = build_interaction_matrix(
        train_df, n_users, n_items, implicit_threshold=settings.implicit_threshold
    )

    # Select model
    if model_type == "random":
        model: Any = RandomRecommender()
        model.fit(implicit_matrix)
    elif model_type == "popular":
        model = PopularityRecommender()
        model.fit(implicit_matrix)
    elif model_type == "als":
        params = {
            "factors": hyperparams.get("factors", settings.als_factors),
            "regularization": hyperparams.get("regularization", settings.als_regularization),
            "iterations": hyperparams.get("iterations", settings.als_iterations),
            "alpha": hyperparams.get("alpha", settings.als_alpha),
        }
        model = ALSRecommender(**params)
        model.fit(implicit_matrix)
    elif model_type == "content_based":
        from src.features.item_features import build_item_features

        item_feat_df = build_item_features(train_df, movies_df)
        genre_cols = [
            c
            for c in item_feat_df.columns
            if c not in ("avg_rating", "num_ratings", "popularity_score", "year")
            and not c.startswith("tag_tfidf_")
        ]
        item_features_array = item_feat_df[genre_cols].values.astype(np.float32)
        model = ContentBasedRecommender()
        model.fit(explicit_matrix, item_features=item_features_array)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate on validation set
    log.info("Evaluating model on validation set...")
    k = settings.top_k
    metrics = evaluate_model(
        model=model,
        test_ratings=val_df,
        train_interaction=implicit_matrix,
        k=k,
        threshold=settings.implicit_threshold,
    )

    # Compute coverage
    all_recs: list[list[int]] = []
    sample_users = val_df["user_idx"].unique()[:500]
    for uid in sample_users:
        if uid < implicit_matrix.shape[0]:
            recs = model.recommend(int(uid), n=k, exclude_seen=True)
            all_recs.append([item_id for item_id, _ in recs])
    metrics["coverage"] = coverage(all_recs, n_items)

    log.info("Validation metrics: %s", metrics)
    return model, metrics


def run_training_pipeline(model_type: str = "als") -> None:
    """Full training pipeline with MLflow logging."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    hyperparams: dict[str, Any] = {}
    if model_type == "als":
        hyperparams = {
            "factors": settings.als_factors,
            "regularization": settings.als_regularization,
            "iterations": settings.als_iterations,
            "alpha": settings.als_alpha,
        }

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset_version", settings.dataset_name)
        mlflow.set_tag("split_strategy", "temporal")

        # Log hyperparameters
        mlflow.log_params(hyperparams if hyperparams else {"model_type": model_type})

        # Train and evaluate
        start = time.time()
        model, metrics = train_and_evaluate(model_type=model_type, hyperparams=hyperparams)
        train_time = time.time() - start

        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", train_time)

        # Save and log model artifact
        artifacts_dir = settings.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "als":
            artifact = {
                "model_type": model_type,
                "user_factors": model.user_factors,
                "item_factors": model.item_factors,
                "n_users": model.user_factors.shape[0],
                "n_items": model.item_factors.shape[0],
                "hyperparams": hyperparams,
                "metrics": metrics,
                "model_version": settings.model_version,
            }
        elif model_type == "content_based":
            item_features = model.item_features
            user_profiles = model.user_profiles
            artifact = {
                "model_type": model_type,
                "item_features": item_features,
                "user_profiles": user_profiles,
                "n_users": user_profiles.shape[0] if user_profiles is not None else 0,
                "n_items": item_features.shape[0] if item_features is not None else 0,
                "hyperparams": hyperparams,
                "metrics": metrics,
                "model_version": settings.model_version,
            }
        else:
            artifact = {
                "model_type": model_type,
                "metrics": metrics,
                "model_version": settings.model_version,
            }

        model_path = artifacts_dir / "model.pkl"
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path))

        # Write sibling metadata so the API can report which model it serves.
        meta = {
            "model_type": model_type,
            "model_version": settings.model_version,
            "mlflow_run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA"),
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        }
        meta_path = artifacts_dir / "model_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
        mlflow.log_artifact(str(meta_path))

        log.info(
            "Training complete. Run ID: %s, Metrics: %s",
            run.info.run_id,
            metrics,
        )

    log.info("Model saved to %s (meta: %s)", model_path, meta_path)


def main() -> None:
    """Entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    import argparse

    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument(
        "--model",
        type=str,
        default="als",
        choices=["random", "popular", "als", "content_based"],
        help="Model type to train",
    )
    args = parser.parse_args()

    run_training_pipeline(model_type=args.model)


if __name__ == "__main__":
    main()
