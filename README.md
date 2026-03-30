# Movie Recommendation System

Hệ thống gợi ý phim end-to-end — từ data pipeline, model training, REST API đến monitoring.

**Use case:** User mở app → nhận danh sách "Recommended for you" dựa trên lịch sử rating.

---

## Tech Stack

| Layer | Công nghệ |
|---|---|
| Language | Python 3.11+ |
| ML | `implicit` (ALS), scikit-learn, SciPy |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Data | Pandas, NumPy, Parquet |
| Container | Docker + docker-compose |
| Monitoring | Prometheus + Grafana |
| Testing / Lint | pytest, ruff |
| CI/CD | GitHub Actions |

---

## Cài đặt

```bash
git clone <repo-url>
cd movie-recommendation-system

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r requirements-test.txt
cp .env.example .env            # tuỳ chỉnh nếu cần
```

---

## Dữ liệu

Đặt `movies.csv` và `ratings.csv` vào thư mục `data/raw/`, sau đó chạy:

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

# Train tất cả để so sánh
make train-all

# Xem kết quả trên MLflow UI
make mlflow-ui          # mở http://localhost:5000
```

Model artifact được lưu vào `artifacts/model.pkl`.

### Benchmark (ml-latest-small)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Random | ~0.01 | ~0.005 | ~0.01 |
| Popularity | ~0.05 | ~0.03 | ~0.04 |
| ALS (factors=100) | ~0.10–0.15 | ~0.06–0.10 | ~0.10–0.15 |

---

## API

```bash
make serve
# hoặc
uvicorn src.serving.app:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

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

## Full Stack (Docker)

```bash
make docker-up          # build + start tất cả services
docker-compose logs -f api
make docker-down
```

| Service | URL | Thông tin đăng nhập |
|---|---|---|
| API | http://localhost:8000 | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## Tests

```bash
make test                               # toàn bộ tests + coverage
pytest tests/ -x -q                     # dừng tại lỗi đầu tiên
pytest tests/test_api.py -v             # chỉ API tests
```

44 tests — data pipeline, models, evaluation metrics, API endpoints.

---

## Lint

```bash
ruff check src/ tests/ --fix    # lint + auto-fix
ruff format src/ tests/         # format
make lint-check                 # CI mode (không sửa)
```

---

## Cấu trúc Project

```
├── data/
│   ├── raw/               # CSV đầu vào (không commit)
│   └── processed/         # Parquet sau preprocessing (không commit)
├── src/
│   ├── config.py          # Settings tập trung (pydantic-settings)
│   ├── data/
│   │   ├── ingestion.py       # Download & validate dataset
│   │   └── preprocessing.py  # Filter → encode → split → save
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
│   │   └── train.py           # Pipeline training + MLflow logging
│   └── serving/
│       ├── app.py             # FastAPI app
│       ├── recommender.py     # Model loader + inference
│       └── schemas.py         # Pydantic schemas
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboard.json
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Biến môi trường

Xem [.env.example](.env.example). Các biến thường dùng:

```bash
RECSYS_ALS_FACTORS=100
RECSYS_ALS_ITERATIONS=30
RECSYS_ALS_ALPHA=40.0
RECSYS_IMPLICIT_THRESHOLD=3.5
RECSYS_MODEL_VERSION=als_v1
RECSYS_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

---

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`):

1. **test** — ruff lint + pytest + upload coverage
2. **build** — Docker build + smoke test
3. **deploy** *(main only)* — push image lên `ghcr.io`

Cần set secret `REGISTRY_TOKEN` trong GitHub repo settings.
