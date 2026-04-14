# Movie Recommendation System

Hệ thống gợi ý phim end-to-end — từ data pipeline, model training, REST API đến MLOps monitoring đầy đủ.

**Use case:** User mở app → nhận danh sách "Recommended for you" dựa trên lịch sử rating.

---

## Tech Stack

| Layer | Công nghệ |
|---|---|
| Language | Python 3.11+ |
| ML | `implicit` (ALS), scikit-learn, SciPy |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow (PostgreSQL + MinIO backend) |
| Orchestration | Apache Airflow (LocalExecutor) |
| Drift Detection | Evidently AI |
| Metrics | Prometheus + Grafana |
| Log Shipping | Filebeat → Elasticsearch → Kibana |
| Object Storage | MinIO (S3-compatible) |
| Database | PostgreSQL 16 (MLflow + Airflow metadata) |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions (self-hosted runner, Mac arm64) |
| Testing / Lint | pytest, ruff, mypy |

---

## Quick Start (Local)

```bash
git clone <repo-url>
cd DDM501-Movie-Recommendation-System

cp .env.example .env        # đổi mọi "changeme" trong .env trước khi chạy

docker compose -f docker-compose.prod.yml up -d
```

Để bỏ Airflow (nhẹ hơn khi dev):

```bash
docker compose -f docker-compose.prod.yml up -d \
  --scale airflow-webserver=0 --scale airflow-scheduler=0
```

### Service URLs

| Service | URL | Đăng nhập |
|---|---|---|
| API | http://localhost:8000 | — |
| API Docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | `MINIO_ROOT_USER` / `MINIO_PASSWORD` |
| Grafana | http://localhost:3000 | admin / `GRAFANA_PASSWORD` |
| Kibana | http://localhost:5601 | — |
| Prometheus | http://localhost:9090 | — |
| Airflow | http://localhost:8080 | admin / `AIRFLOW_ADMIN_PASSWORD` |
| Evidently | http://localhost:8001 | — |

> **Remote server:** Dùng SSH tunnel trước:
> ```bash
> ssh -L 5000:localhost:5000 -L 3000:localhost:3000 -L 5601:localhost:5601 -L 8080:localhost:8080 user@host
> ```

---

## Local Development (không Docker)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r requirements-test.txt
cp .env.example .env
```

---

## Dữ liệu

Đặt `movies.csv` và `ratings.csv` vào `data/raw/`, sau đó chạy:

```bash
python -m src.data.preprocessing
```

Pipeline sẽ:
- Lọc cold-start (user < 20 ratings, movie < 10 ratings)
- Encode `userId`/`movieId` → index nguyên liên tục
- Temporal split 70 / 15 / 15
- Lưu Parquet vào `data/processed/`

**Schema đầu vào bắt buộc:**

| File | Cột |
|---|---|
| `ratings.csv` | `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` |
| `movies.csv` | `movieId`, `title`, `genres` (pipe-separated) |

> `tags.csv` là tuỳ chọn — bỏ qua nếu không có.

---

## Training

```bash
# Train ALS + log lên MLflow
python -m src.training.train --model als

# Các model khác
python -m src.training.train --model popular
python -m src.training.train --model content_based

# Xem kết quả
open http://localhost:5000   # MLflow UI
```

Model artifact lưu vào `artifacts/model.pkl` (và `~/recsys-artifacts/` khi chạy qua CI).

### Benchmark (ml-latest-small)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Random | ~0.01 | ~0.005 | ~0.01 |
| Popularity | ~0.05 | ~0.03 | ~0.04 |
| ALS (factors=100) | ~0.10–0.15 | ~0.06–0.10 | ~0.10–0.15 |

---

## API

```bash
uvicorn src.serving.app:app --reload --port 8000
```

### Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Service status + uptime |
| `POST` | `/recommend` | Top-K gợi ý cho 1 user |
| `POST` | `/recommend/batch` | Top-K cho nhiều users |
| `GET` | `/similar/{movie_id}` | Phim tương tự |
| `GET` | `/metrics` | Prometheus metrics |

**Ví dụ:**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

```json
{
  "user_id": 1,
  "recommendations": [
    {"movie_id": 318, "title": "Shawshank Redemption, The (1994)", "score": 0.95, "genres": ["Crime", "Drama"]},
    {"movie_id": 858, "title": "Godfather, The (1972)", "score": 0.91, "genres": ["Crime", "Drama"]}
  ],
  "model_version": "als_v1",
  "latency_ms": 14.2
}
```

---

## Tests

```bash
pytest tests/ -v --cov=src --cov-fail-under=80   # toàn bộ + coverage
pytest tests/ -x -q                               # dừng tại lỗi đầu tiên
pytest tests/test_api.py -v                       # chỉ API tests
```

---

## Lint

```bash
ruff check src/ tests/ --fix    # lint + auto-fix
ruff format src/ tests/         # format
mypy src/                       # type check
```

---

## Automated Retraining

Hệ thống có hai cách trigger retrain:

### 1. GitHub Actions (train.yml)
- **Tự động:** mỗi Chủ nhật 02:00 UTC
- **Thủ công:** GitHub UI → Actions → Train → Run workflow

Workflow sẽ: download data → preprocess → train → copy model vào container đang chạy → restart API.

### 2. Airflow DAG
- UI: http://localhost:8080 (admin / `AIRFLOW_ADMIN_PASSWORD`)
- DAG `retrain_pipeline` trigger pipeline training nội bộ trong Docker network
- Phù hợp để schedule theo giờ/ngày hoặc trigger theo sự kiện drift

---

## CI/CD

Ba GitHub Actions workflows:

| Workflow | Trigger | Mô tả |
|---|---|---|
| `ci.yml` | push `main`/`develop`, PR | lint → test (≥80% coverage) → Docker build + smoke test → push image lên `ghcr.io` |
| `deploy.yml` | sau khi CI pass trên `main` (hoặc manual) | pull image → `docker compose up` → health check → auto-rollback nếu fail |
| `train.yml` | mỗi Chủ nhật 02:00 UTC (hoặc manual) | download → preprocess → train → hot-reload model vào API |

**Image registry:** `ghcr.io/<org>/recsys-api` (arm64, build native trên self-hosted Mac M-series)

**Secrets cần set trong GitHub repo:**

```
GRAFANA_PASSWORD, POSTGRES_PASSWORD, MINIO_ROOT_USER, MINIO_PASSWORD,
AIRFLOW_ADMIN_PASSWORD, AIRFLOW_FERNET_KEY, AIRFLOW_SECRET_KEY
```

> Xem [RUNNER_SETUP.md](RUNNER_SETUP.md) để cài đặt self-hosted runner.

---

## Cấu trúc Project

```
├── src/
│   ├── config.py              # Settings tập trung (pydantic-settings)
│   ├── data/
│   │   ├── ingestion.py       # Download & validate dataset
│   │   └── preprocessing.py  # Filter → encode → split → Parquet
│   ├── features/
│   │   ├── user_features.py   # avg_rating, genre_preference, recency
│   │   ├── item_features.py   # popularity, genres, TF-IDF
│   │   └── interaction.py     # User-Item sparse CSR matrix
│   ├── models/
│   │   ├── baseline.py        # Random, Popularity
│   │   ├── collaborative.py   # ALS (implicit)
│   │   └── content_based.py   # TF-IDF cosine similarity
│   ├── evaluation/
│   │   └── metrics.py         # Precision/Recall/NDCG@K, Coverage, Diversity
│   ├── training/
│   │   └── train.py           # Training pipeline + MLflow logging
│   └── serving/
│       ├── app.py             # FastAPI app
│       ├── recommender.py     # Model loader + inference
│       └── schemas.py         # Pydantic schemas
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── monitoring/
│   ├── airflow/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── dags/retrain_dag.py
│   ├── evidently/             # Drift detection service
│   ├── filebeat/
│   │   └── filebeat.yml       # Ships logs → Elasticsearch
│   ├── grafana/
│   │   └── provisioning/      # Auto-provisioned dashboards + datasources
│   ├── logstash/              # Kept for reference (không dùng)
│   ├── mlflow/Dockerfile
│   ├── postgres/
│   │   └── init-multiple-dbs.sh
│   └── prometheus.yml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── scripts/
│   └── setup_runner.sh        # Script cài self-hosted runner
├── .github/workflows/
│   ├── ci.yml
│   ├── deploy.yml
│   └── train.yml
├── Dockerfile
├── docker-compose.yml         # Dev stack (API + MLflow nhẹ)
├── docker-compose.prod.yml    # Full production stack
├── pyproject.toml
├── requirements.txt
├── requirements-serve.txt
├── requirements-test.txt
├── .env.example
└── RUNNER_SETUP.md
```

---

## Biến môi trường

Xem [.env.example](.env.example) — copy và đổi tất cả `changeme` trước khi chạy.

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `POSTGRES_PASSWORD` | Có | Dùng chung cho MLflow + Airflow |
| `MINIO_ROOT_USER` | Có | MinIO admin username |
| `MINIO_PASSWORD` | Có | MinIO admin password |
| `GRAFANA_PASSWORD` | Có | Grafana admin password |
| `AIRFLOW_ADMIN_PASSWORD` | Có | Airflow web UI |
| `AIRFLOW_FERNET_KEY` | Có | Mã hoá connections trong DB |
| `AIRFLOW_SECRET_KEY` | Có | Session secret cho webserver |
| `API_IMAGE` | Không | Pre-built image (bỏ trống để build từ Dockerfile) |
