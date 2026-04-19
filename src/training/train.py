"""Pipeline huấn luyện LightFM Hybrid với MLflow experiment tracking.

Quy trình:
1. Đọc ``ratings.csv`` + ``movies.csv`` từ ``data/raw/``.
2. Build ``LightFMBundle`` (split 80/20 random, interactions, item features).
3. Train ``LightFMRecommender`` (WARP, 64 components, lr=0.05, 30 epochs).
4. Đánh giá bằng ``lightfm.evaluation`` (precision@k, recall@k, auc_score).
5. Log mọi tham số / metric vào MLflow; lưu artifact ``model.pkl`` +
   ``model_meta.json`` cho serving.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow
import numpy as np
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

from src.config import settings
from src.features.lightfm_dataset import LightFMBundle, build_lightfm_dataset
from src.models.baseline import PopularityRecommender, RandomRecommender
from src.models.lightfm_hybrid import LightFMRecommender

log = logging.getLogger(__name__)


def _evaluate_lightfm(
    model: LightFMRecommender,
    bundle: LightFMBundle,
    k: int,
) -> dict[str, float]:
    """Đánh giá LightFM trên test split bằng API chuẩn của thư viện."""
    lightfm_model = model.model
    item_features = bundle.item_features
    num_threads = settings.lightfm_num_threads

    # ``train_interactions`` được truyền để loại các cặp train ra khỏi test
    # evaluation — đây là cách chuẩn để tránh data leakage.
    precision = float(
        precision_at_k(
            lightfm_model,
            test_interactions=bundle.test_interactions,
            train_interactions=bundle.train_interactions,
            k=k,
            item_features=item_features,
            num_threads=num_threads,
        ).mean()
    )
    recall = float(
        recall_at_k(
            lightfm_model,
            test_interactions=bundle.test_interactions,
            train_interactions=bundle.train_interactions,
            k=k,
            item_features=item_features,
            num_threads=num_threads,
        ).mean()
    )
    auc = float(
        auc_score(
            lightfm_model,
            test_interactions=bundle.test_interactions,
            train_interactions=bundle.train_interactions,
            item_features=item_features,
            num_threads=num_threads,
        ).mean()
    )

    # In kết quả rõ ràng như người dùng yêu cầu.
    print("=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ LIGHTFM HYBRID (k={k})")
    print("=" * 60)
    print(f"  Precision@{k}: {precision:.4f}")
    print(f"  Recall@{k}   : {recall:.4f}")
    print(f"  AUC score    : {auc:.4f}")
    print("=" * 60)

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "auc": auc,
        # F1 tiện log cho Prometheus/MLflow.
        "f1_score": (
            0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        ),
        "n_users_evaluated": float(bundle.test_interactions.shape[0]),
    }


def _build_user_seen(train_interactions: Any) -> dict[int, set[int]]:
    """Lấy dict user_idx -> set(item_idx) đã xem từ matrix train."""
    csr = train_interactions.tocsr()
    seen: dict[int, set[int]] = {}
    for uid in range(csr.shape[0]):
        start, end = int(csr.indptr[uid]), int(csr.indptr[uid + 1])
        seen[uid] = {int(x) for x in csr.indices[start:end]}
    return seen


def train_and_evaluate(
    model_type: str = "lightfm",
) -> tuple[Any, dict[str, float], LightFMBundle | None]:
    """Train + evaluate; trả về (model, metrics, bundle).

    Bundle chỉ có giá trị với ``lightfm``; baselines trả về None.
    """
    if model_type == "lightfm":
        bundle = build_lightfm_dataset(
            ratings_csv=settings.data_raw_dir / "ratings.csv",
            movies_csv=settings.data_raw_dir / "movies.csv",
            test_size=settings.lightfm_test_size,
            seed=settings.lightfm_split_seed,
        )

        model = LightFMRecommender(
            no_components=settings.lightfm_no_components,
            loss=settings.lightfm_loss,
            learning_rate=settings.lightfm_learning_rate,
            epochs=settings.lightfm_epochs,
            num_threads=settings.lightfm_num_threads,
            random_state=settings.lightfm_split_seed,
        )
        model.fit(bundle.train_interactions, item_features=bundle.item_features)

        metrics = _evaluate_lightfm(model, bundle, k=settings.top_k)
        return model, metrics, bundle

    # --- Baselines (smoke-test cho CI, không cần item features) --------------
    # Dùng lại bundle để có cùng matrix interactions — tránh chệch so sánh.
    bundle = build_lightfm_dataset(
        ratings_csv=settings.data_raw_dir / "ratings.csv",
        movies_csv=settings.data_raw_dir / "movies.csv",
        test_size=settings.lightfm_test_size,
        seed=settings.lightfm_split_seed,
    )
    interactions = bundle.train_interactions

    if model_type == "random":
        baseline: Any = RandomRecommender()
    elif model_type == "popular":
        baseline = PopularityRecommender()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    baseline.fit(interactions)

    # Baselines chỉ cần AUC-less rough metric — placeholder để MLflow vẫn log.
    metrics = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "auc": 0.0,
        "f1_score": 0.0,
        "n_users_evaluated": float(interactions.shape[0]),
    }
    return baseline, metrics, bundle


def _build_artifact(
    model_type: str,
    model: Any,
    bundle: LightFMBundle | None,
    metrics: dict[str, float],
    hyperparams: dict[str, Any],
) -> dict[str, Any]:
    """Dựng dict artifact để joblib dump cho serving load."""
    if model_type == "lightfm" and bundle is not None:
        return {
            "model_type": "lightfm",
            "model": model.model,  # lightfm.LightFM object
            "item_features": bundle.item_features,
            "item_embeddings": np.asarray(model.item_embeddings),
            "user_id_map": bundle.user_id_map,
            "item_id_map": bundle.item_id_map,
            "reverse_item_id_map": bundle.reverse_item_id_map,
            "reverse_user_id_map": bundle.reverse_user_id_map,
            "user_seen": _build_user_seen(bundle.train_interactions),
            "movies": bundle.movies,
            "n_users": bundle.train_interactions.shape[0],
            "n_items": bundle.train_interactions.shape[1],
            "hyperparams": hyperparams,
            "metrics": metrics,
            "model_version": settings.model_version,
            "num_threads": settings.lightfm_num_threads,
        }

    # Baselines: serving không bắt buộc chạy baseline; lưu metadata tối thiểu.
    return {
        "model_type": model_type,
        "metrics": metrics,
        "hyperparams": hyperparams,
        "model_version": settings.model_version,
    }


def run_training_pipeline(model_type: str = "lightfm") -> None:
    """Chạy full pipeline: train → evaluate → log MLflow → dump artifact."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    hyperparams: dict[str, Any] = {}
    if model_type == "lightfm":
        hyperparams = {
            "no_components": settings.lightfm_no_components,
            "loss": settings.lightfm_loss,
            "learning_rate": settings.lightfm_learning_rate,
            "epochs": settings.lightfm_epochs,
            "num_threads": settings.lightfm_num_threads,
            "test_size": settings.lightfm_test_size,
            "split_seed": settings.lightfm_split_seed,
        }

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset_version", "ml-latest-small")
        mlflow.set_tag("split_strategy", "random_80_20")

        mlflow.log_params(hyperparams if hyperparams else {"model_type": model_type})

        start = time.time()
        model, metrics, bundle = train_and_evaluate(model_type=model_type)
        train_time = time.time() - start

        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", train_time)

        # --- Lưu artifact ----------------------------------------------------
        artifacts_dir = settings.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact = _build_artifact(model_type, model, bundle, metrics, hyperparams)

        model_path = artifacts_dir / "model.pkl"
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path))

        # Metadata nhẹ để /health của API đọc nhanh không cần load model.
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
            "Training xong. Run ID: %s, metrics: %s",
            run.info.run_id,
            metrics,
        )

    log.info("Saved artifact %s (meta: %s)", model_path, meta_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train LightFM recommendation model")
    parser.add_argument(
        "--model",
        type=str,
        default="lightfm",
        choices=["lightfm", "popular", "random"],
        help="Model type to train (lightfm là model chính; baselines chỉ để smoke-test)",
    )
    args = parser.parse_args()
    run_training_pipeline(model_type=args.model)


if __name__ == "__main__":
    main()
