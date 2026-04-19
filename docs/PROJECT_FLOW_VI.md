# PROJECT_FLOW_VI — Phân tích chi tiết DDM501 Movie Recommendation System

Tài liệu này mô tả **luồng chạy end-to-end**, **function nào xử lý việc gì**, và **monitoring được wiring ra sao** cho dự án `DDM501-Movie-Recommendation-System`.

> File tham chiếu tiếng Anh (high-level): [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md), [`docs/OPERATIONS.md`](./OPERATIONS.md), [`docs/DEVELOPMENT.md`](./DEVELOPMENT.md), [`docs/LIGHTFM_ALGORITHM.md`](./LIGHTFM_ALGORITHM.md), [`docs/MODELS.md`](./MODELS.md). Tài liệu này tập trung vào **luồng thực thi** và **ai gọi ai**, bổ sung cho các file kia.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Stack công nghệ & vai trò từng service](#2-stack-công-nghệ--vai-trò-từng-service)
3. [Luồng chạy chi tiết](#3-luồng-chạy-chi-tiết)
   - 3.1. [Preprocessing](#31-luồng-1--preprocessing)
   - 3.2. [Training LightFM Hybrid](#32-luồng-2--training-lightfm-hybrid)
   - 3.3. [Serving realtime (FastAPI)](#33-luồng-3--serving-realtime-fastapi)
   - 3.4. [Airflow weekly retrain](#34-luồng-4--airflow-weekly-retrain)
   - 3.5. [CI/CD (GitHub Actions)](#35-luồng-5--cicd-github-actions)
4. [Function / class reference](#4-function--class-reference)
5. [Monitoring — như thế nào?](#5-monitoring--như-thế-nào)
6. [Invariant train/serve contract](#6-invariant-trainserve-contract)
7. [Cheatsheet vận hành](#7-cheatsheet-vận-hành)
8. [Quick reference paths](#8-quick-reference-paths)

---

## 1. Tổng quan hệ thống

Đây là hệ thống gợi ý phim hybrid trên dataset **MovieLens (ml-latest-small)**, dùng **LightFM với loss WARP** (collaborative filtering + genre content features). Serving qua **FastAPI**, tracking experiment qua **MLflow (Postgres + MinIO)**, retrain định kỳ bằng **Airflow**, observability gồm **Prometheus + Grafana + Evidently + ELK (Elasticsearch/Kibana/Filebeat)**, CI/CD qua **GitHub Actions** (self-hosted Mac ARM64 runner).

```
                 ┌────────────────────────────────────────────────────────────┐
                 │                    DATA PIPELINE                           │
                 │                                                            │
   data/raw/  ──►│  preprocess  ──►  data/processed/*.parquet                 │
   (CSV)         │   (src/data)                                               │
                 │        │                                                   │
                 │        └──► build_lightfm_dataset ──► LightFMBundle        │
                 │               (src/features)                               │
                 └─────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
                 ┌────────────────────────────────────────────────────────────┐
                 │                    TRAINING PIPELINE                       │
                 │                                                            │
                 │  LightFMRecommender.fit (WARP, 30 epochs)                  │
                 │          │                                                 │
                 │          ├──► MLflow (params + metrics + artifact)         │
                 │          │                                                 │
                 │          └──► $RECSYS_ARTIFACTS_DIR/                       │
                 │                 ├── model.pkl                              │
                 │                 └── model_meta.json                        │
                 └─────────────────────────┬──────────────────────────────────┘
                                           │  bind-mount
                                           ▼
                 ┌────────────────────────────────────────────────────────────┐
                 │                    SERVING (FastAPI)                       │
                 │                                                            │
   client ──► POST /recommend ───────► RecommenderService.recommend           │
   client ──► POST /recommend/batch                                           │
   client ──► GET  /similar/{id}  ───► RecommenderService.similar_items       │
   client ──► GET  /health                                                    │
   Prom.  ──► GET  /metrics                                                   │
                 └──────────────────────────┬─────────────────────────────────┘
                                            │
         ┌──────────────────┬────────────────┼─────────────────┬────────────┐
         ▼                  ▼                ▼                 ▼            ▼
    Prometheus          Filebeat          Evidently         Grafana     cAdvisor
    (metrics)       (ES + Kibana)      (data drift)       (dashboard)  (container)
```

5 endpoint public của API: `/recommend`, `/recommend/batch`, `/similar/{movie_id}`, `/health`, `/metrics`.

---

## 2. Stack công nghệ & vai trò từng service

Stack đầy đủ định nghĩa ở [`docker-compose.prod.yml`](../docker-compose.prod.yml).

| Service | Image / Framework | Port | Trách nhiệm | File cấu hình |
|---|---|---|---|---|
| `api` | FastAPI 0.x + uvicorn | 8000 | Phục vụ inference realtime LightFM | [`Dockerfile`](../Dockerfile), [`src/serving/app.py`](../src/serving/app.py) |
| `mlflow` | `mlflow server` | 5000 | Experiment tracking, artifact proxy (--serve-artifacts) | [`monitoring/mlflow/Dockerfile`](../monitoring/mlflow) |
| `postgres` | `postgres:16-alpine` | 5432 | Backend store cho MLflow & Airflow | [`monitoring/postgres/init-multiple-dbs.sh`](../monitoring/postgres/init-multiple-dbs.sh) |
| `minio` | `minio/minio` | 9000/9001 | S3-compatible artifact store cho MLflow (`bucket mlflow-artifacts`) | — |
| `evidently` | FastAPI + APScheduler + Evidently AI | 8001 | Tính data drift, expose Prometheus gauges | [`monitoring/evidently/app.py`](../monitoring/evidently/app.py) |
| `prometheus` | `prom/prometheus` | 9090 | Scrape metrics, evaluate alert rules | [`monitoring/prometheus.yml`](../monitoring/prometheus.yml), [`monitoring/prometheus-rules.yml`](../monitoring/prometheus-rules.yml) |
| `grafana` | `grafana/grafana` | 3000 | Visualize metrics, auto-provision dashboards | [`monitoring/grafana/`](../monitoring/grafana) |
| `cadvisor` | `gcr.io/cadvisor/cadvisor` | 8081 | Container CPU/mem/network | — |
| `node-exporter` | `prom/node-exporter` | 9100 | Host CPU/RAM/disk | — |
| `elasticsearch` | 8.13.0 | 9200 | Log store | — |
| `kibana` | 8.13.0 | 5601 | Log search UI, auto-load saved objects | [`monitoring/kibana/init.sh`](../monitoring/kibana/init.sh) |
| `filebeat` | 8.13.0 | — | Ship container JSON log lên Elasticsearch | [`monitoring/filebeat/filebeat.yml`](../monitoring/filebeat/filebeat.yml) |
| `airflow-webserver` / `airflow-scheduler` | Airflow (LocalExecutor) | 8080 | Orchestrate retrain DAG `recsys_weekly_retrain` | [`monitoring/airflow/dags/retrain_dag.py`](../monitoring/airflow/dags/retrain_dag.py) |

Phân tầng:

- **Core runtime** → `api`, `mlflow`, `postgres`, `minio`
- **Monitoring** → `prometheus`, `grafana`, `evidently`, `cadvisor`, `node-exporter`
- **Logging** → `elasticsearch`, `kibana`, `kibana-init`, `filebeat`
- **Orchestration** → `airflow-init`, `airflow-webserver`, `airflow-scheduler`

Tất cả chạy trên network `backend`, các UI (trừ API 8000) bind `127.0.0.1` — an toàn khi chạy trên VPS, cần SSH tunnel để truy cập.

---

## 3. Luồng chạy chi tiết

### 3.1. Luồng 1 — Preprocessing

**File:** [`src/data/preprocessing.py`](../src/data/preprocessing.py)
**Trigger:**
- Thủ công: `python -m src.data.preprocessing` (hoặc `make data`)
- Tự động: Airflow DAG task `preprocess_data` nếu chưa có `train.parquet`
- CI/CD: step "Preprocess data" trong [`train.yml`](../.github/workflows/train.yml)

**Trình tự hàm gọi:**

```
preprocess()
  ├─ load_raw_data(raw_dir)
  │    └─ load_raw_csv(raw_dir, "ratings.csv" | "movies.csv" | "tags.csv")
  ├─ filter_cold_start(ratings, min_user=20, min_item=10)  # lặp đến khi hội tụ
  ├─ encode_ids(ratings_filtered)
  │    └─ build_id_mapping(series)  # user & movie
  ├─ temporal_split(ratings_encoded, 0.70, 0.15)  # sort theo timestamp
  ├─ save_parquet(train | val | test | user_id_map | movie_id_map | movies | tags)
  └─ compute_sparsity(n_ratings, n_users, n_items)
```

**Output** (đặt trong `$RECSYS_DATA_PROCESSED_DIR`, mặc định `./data/processed/`):

- `train.parquet` / `val.parquet` / `test.parquet` (phân chia theo thời gian 70/15/15)
- `user_id_map.parquet`, `movie_id_map.parquet` (encode ID thành integer liên tục 0-based)
- `movies.parquet` (đã merge `movie_idx`, phim đã filter)
- `tags.parquet` (user+movie đều phải sống sót sau filter)

**Return:** `SplitStats(train_size, val_size, test_size, n_users, n_items, sparsity)`

**Lưu ý quan trọng**: bộ parquet này là kết quả của **temporal split 70/15/15**, nhưng training LightFM (xem 3.2) **không đọc parquet** mà đọc thẳng `ratings.csv` + `movies.csv` và tự split random 80/20 seed=42. Parquet phục vụ:
- Evidently lấy `train.parquet` làm reference để tính drift
- Downstream analytics / notebook
- (Tạm thời không phục vụ training LightFM)

---

### 3.2. Luồng 2 — Training LightFM Hybrid

**File:** [`src/training/train.py`](../src/training/train.py) + [`src/features/lightfm_dataset.py`](../src/features/lightfm_dataset.py) + [`src/models/lightfm_hybrid.py`](../src/models/lightfm_hybrid.py)
**Trigger:**
- `python -m src.training.train --model lightfm` (hoặc `make train`)
- Airflow task `train_model` (cron `0 2 * * 0`)
- GitHub Actions `train.yml` (cron `0 2 * * 0` + `workflow_dispatch`)

**Trình tự hàm gọi:**

```
main()  →  run_training_pipeline(model_type="lightfm")
             │
             ├─ mlflow.set_tracking_uri / set_experiment
             ├─ mlflow.start_run("lightfm_run")
             │    ├─ set_tag(model_type | dataset_version | split_strategy)
             │    ├─ log_params(hyperparams)
             │    ├─ train_and_evaluate("lightfm")
             │    │    ├─ build_lightfm_dataset(ratings_csv, movies_csv, test_size=0.2, seed=42)
             │    │    │    └─ returns LightFMBundle
             │    │    ├─ LightFMRecommender(no_components=64, loss="warp",
             │    │    │                     lr=0.05, epochs=30, num_threads=4).fit(
             │    │    │       train_interactions, item_features=item_features)
             │    │    │     └─ cache _user_seen + _item_embeddings (get_item_representations)
             │    │    └─ _evaluate_lightfm(model, bundle, k=10)
             │    │          ├─ precision_at_k(..., train_interactions=..., item_features=...)
             │    │          ├─ recall_at_k(...)
             │    │          ├─ auc_score(...)
             │    │          └─ f1 = 2*p*r / (p+r)
             │    │
             │    ├─ mlflow.log_metrics(metrics + training_time_seconds)
             │    ├─ _build_artifact(...)  →  dict với model/item_features/embeddings/map/user_seen/movies/metrics
             │    ├─ joblib.dump(artifact, artifacts_dir / "model.pkl")
             │    ├─ mlflow.log_artifact(model.pkl)
             │    ├─ write model_meta.json (trained_at, mlflow_run_id, git_sha, metrics)
             │    └─ mlflow.log_artifact(model_meta.json)
             └─ (end run)
```

**Hyperparams (đặt trong `src/config.py`):**
- `no_components = 64`
- `loss = "warp"` (Weighted Approximate-Rank Pairwise)
- `learning_rate = 0.05`
- `epochs = 30`
- `num_threads = 4`
- `test_size = 0.2` (random split, seed=42)

**Metric được log:** `precision_at_k`, `recall_at_k`, `auc`, `f1_score`, `n_users_evaluated`, `training_time_seconds`. Khi evaluate có truyền **cả `train_interactions`** vào `precision_at_k / recall_at_k / auc_score` để loại cặp train ra khỏi đánh giá (chống leakage — đây là cách chuẩn của `lightfm.evaluation`).

**Artifact (`model.pkl`) chứa:**

```python
{
  "model_type": "lightfm",
  "model": <lightfm.LightFM>,               # đã fit
  "item_features": <csr_matrix>,            # cần cho predict
  "item_embeddings": <ndarray>,             # cache cho similar_items
  "user_id_map": {orig_userId: idx},
  "item_id_map": {orig_movieId: idx},
  "reverse_item_id_map": {idx: orig_movieId},
  "reverse_user_id_map": {idx: orig_userId},
  "user_seen": {user_idx: set(item_idx)},   # để exclude_seen
  "movies": <DataFrame>,                    # title + genres
  "n_users": ..., "n_items": ...,
  "hyperparams": {...},
  "metrics": {...},
  "model_version": "lightfm_v1",
  "num_threads": 4,
}
```

**`model_meta.json`** (được `/health` đọc không cần load model):

```json
{
  "model_type": "lightfm",
  "model_version": "lightfm_v1",
  "mlflow_run_id": "...",
  "trained_at": "2026-04-19T10:00:00+00:00",
  "git_sha": "...",
  "metrics": { "precision_at_k": ..., "recall_at_k": ..., "auc": ..., ... }
}
```

**Baselines**: `RandomRecommender` và `PopularityRecommender` nằm ở [`src/models/baseline.py`](../src/models/baseline.py), chỉ dùng để smoke-test CI hoặc lower-bound so sánh — **không được serving load**. Nếu `model.pkl` không phải `model_type == "lightfm"`, serving raise `RecommenderServiceError`.

---

### 3.3. Luồng 3 — Serving realtime (FastAPI)

**File gốc:** [`src/serving/app.py`](../src/serving/app.py)
**Entry point container:** `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`

#### 3.3.1. Startup — `lifespan()`

```
@asynccontextmanager lifespan(app)
  ├─ state.reset_start_time()                          # dùng cho /health uptime
  ├─ configure_logging(settings.log_level)             # JSON formatter + bắt uvicorn loggers
  ├─ state.service.load(model_path, data_processed_dir)
  │     ├─ joblib.load(model.pkl)
  │     ├─ check model_type == "lightfm" (else raise)
  │     ├─ unpack model, item_features, item_embeddings, các mapping
  │     ├─ build _movie_meta (title + genres)
  │     ├─ _compute_popular_items(top_k=100)            # fallback cold-start
  │     └─ load model_meta.json (trained_at, run_id, git_sha)
  ├─ yield                                              # app sẵn sàng nhận request
  └─ log "service_shutdown"
```

Nếu load fail → log `model_load_failed` + trả `/health` status `unavailable`, các endpoint `/recommend` trả **503 "Model not loaded"**.

#### 3.3.2. Middleware

Có 2 middleware (thứ tự: CORS trong cùng → Correlation outer):

1. **`CorrelationIdMiddleware`** ([`app.py:80`](../src/serving/app.py#L80))
   - Lấy `X-Request-ID` từ header, hoặc sinh UUID4.
   - Gắn vào `request.state.request_id` để router đọc.
   - Đo `latency_ms = (time.perf_counter() - start) * 1000`.
   - Emit log `request_completed` (hoặc `request_error`) JSON có: request_id, method, path, status_code, latency_ms, client_ip, user_agent.
   - Gắn lại header `X-Request-ID` trên response.
2. **`CORSMiddleware`** — allow `*`, methods `GET|POST`, credentials off.

#### 3.3.3. Endpoint 1 — `POST /recommend`

File: [`src/serving/routers/recommend.py`](../src/serving/routers/recommend.py)

Request body (Pydantic `RecommendRequest`):
```json
{"user_id": 1, "top_k": 10, "exclude_seen": true}
```

Flow:

```
recommend(http_request, request)
  ├─ is_cold_start = not state.service.is_known_user(request.user_id)
  ├─ results = state.service.recommend(user_id, top_k, exclude_seen)
  │     ├─ check _recommend_cache[(uid, k, excl)]  # LRU size 1024
  │     ├─ user_idx = _user_to_idx[user_id]
  │     │     └─ None? → trả _popular_items[:top_k]     (cold-start fallback)
  │     ├─ scores = model.predict(user_idx, item_ids, item_features, num_threads)
  │     ├─ if exclude_seen: scores[seen_idx] = -inf
  │     ├─ top_indices = argpartition(-scores, top_k)
  │     └─ [_format_item(idx, scores[idx]) for idx in top_indices]
  │             └─ score = sigmoid(raw) = 1 / (1 + exp(-raw))  # chuẩn hóa về [0, 1]
  │
  ├─ if is_cold_start: COLD_START_COUNTER.inc()
  ├─ LATENCY_HISTOGRAM.labels(endpoint="/recommend").observe(latency_s)
  ├─ REQUEST_COUNTER.labels(status="success", endpoint="/recommend").inc()
  ├─ log "recommendation_served" (JSON: request_id, user_id, top_k, movie_ids, latency_ms, ...)
  └─ return RecommendResponse(user_id, recommendations, model_version, latency_ms)
```

Response (Pydantic `RecommendResponse`):
```json
{
  "user_id": 1,
  "recommendations": [
    {"movie_id": 318, "title": "Shawshank Redemption (1994)", "score": 0.9123, "genres": ["Crime","Drama"]},
    ...
  ],
  "model_version": "lightfm_v1",
  "latency_ms": 12.345
}
```

Error handling:
- `ModelNotLoadedError` → HTTP 503 + counter `status="error"`
- `Exception` → HTTP 500 + counter `status="error"`

#### 3.3.4. Endpoint 2 — `POST /recommend/batch`

Body:
```json
{"user_ids": [1, 2, 3], "top_k": 10}
```

Lặp qua từng user_id gọi `state.service.recommend(...)`, đếm tổng `cold_start_count`, tính total_latency. Giới hạn `min_length=1, max_length=100` user.

#### 3.3.5. Endpoint 3 — `GET /similar/{movie_id}?top_k=10`

File: [`src/serving/routers/similar.py`](../src/serving/routers/similar.py)

```
similar_items(movie_id, top_k)
  ├─ item_idx = _item_to_idx[movie_id]    (None → UnknownMovieError → 404)
  ├─ target = _item_embeddings[item_idx]
  ├─ sims = (_item_embeddings @ target) / (norms * |target|)   # cosine
  ├─ sims[item_idx] = -inf                                      # loại chính nó
  ├─ top_indices = argpartition(-sims, top_k)
  └─ [_format_item(idx, sims[idx]) for idx in top_indices]
```

Không gọi `model.predict` → **nhanh** vì chỉ nhân ma trận sẵn trong RAM.

#### 3.3.6. Endpoint 4 — `GET /health`

File: [`src/serving/routers/health.py`](../src/serving/routers/health.py)

```
health()
  ├─ uptime = time.monotonic() - state.start_time
  ├─ meta = state.service.model_meta
  └─ return {
       status: "healthy" if loaded else "unavailable",
       model_version, uptime ("1d 2h 30m 15s"),
       model_type, trained_at, mlflow_run_id, git_sha
     }
```

Cả 200 (healthy) lẫn 503 (chưa load model) đều coi là **API start hợp lệ** trong CI (xem 3.5).

#### 3.3.7. Endpoint 5 — `GET /metrics`

File: [`src/serving/routers/prometheus.py`](../src/serving/routers/prometheus.py)

Trả plain-text Prometheus exposition format (`generate_latest()` của `prometheus_client`). Prometheus scrape mỗi **10 giây**.

---

### 3.4. Luồng 4 — Airflow weekly retrain

**File:** [`monitoring/airflow/dags/retrain_dag.py`](../monitoring/airflow/dags/retrain_dag.py)

**DAG:** `recsys_weekly_retrain`, schedule `"0 2 * * 0"` (Chủ nhật 02:00 UTC), `catchup=False`, `retries=1`.

**Task graph:**

```
check_processed_data (BranchPython)
      │
      ├─ (nếu chưa có train.parquet) ──► preprocess_data (Bash: python -m src.data.preprocessing)
      │                                              │
      └─ (đã có)                     ────────────────┼─► train_model (Bash: python -m src.training.train --model lightfm)
                                                     │       trigger_rule="none_failed_min_one_success"
                                                     │
                                                     └─► reload_api (Python: docker SDK restart container)
```

**`reload_api` chi tiết:**

```python
client = docker.from_env()
containers = client.containers.list(filters={"label": "com.docker.compose.service=api"})
if containers:
    containers[0].restart(timeout=30)
```

Airflow container mount `/var/run/docker.sock:ro` nên gọi được Docker SDK để restart service `api`. Không dùng `docker cp` — model mới đã có trong volume bind-mount chung.

**Env của Airflow container (từ `docker-compose.prod.yml` `x-airflow-common`):**
- `RECSYS_DATA_RAW_DIR=/opt/airflow/data/raw`
- `RECSYS_DATA_PROCESSED_DIR=/opt/airflow/data/processed`
- `RECSYS_ARTIFACTS_DIR=/opt/airflow/artifacts`
- `RECSYS_MLFLOW_TRACKING_URI=http://mlflow:5000`
- `PYTHONPATH=/opt/airflow`
- Mount: `./src → /opt/airflow/src:ro`, `./data ↔ /opt/airflow/data`, `${RECSYS_ARTIFACTS_DIR} ↔ /opt/airflow/artifacts`

---

### 3.5. Luồng 5 — CI/CD (GitHub Actions)

3 workflow file trong [`.github/workflows/`](../.github/workflows):

#### 3.5.1. `ci.yml` — Continuous Integration

Trigger: push/PR vào `main`, `develop`.

Jobs tuần tự:

1. **lint** (ubuntu-latest) — `ruff check` + `ruff format --check` trên `src/` + `tests/`.
2. **test** (ubuntu-latest, needs=lint) — cài LightFM từ git master `--no-build-isolation`, cài `requirements*.txt`, chạy `pytest -v --cov=src --cov-fail-under=80` → upload Codecov.
3. **compose-lint** (ubuntu-latest) — `docker compose -f docker-compose.prod.yml config -q` với env dummy để validate syntax.
4. **build** (ubuntu-latest, needs=[test, compose-lint]) — `docker build --target server`, chạy container, `curl /health`. Chấp nhận **200 hoặc 503** (503 = chưa load model, vẫn coi là API start ok).
5. **push** (self-hosted Mac arm64, needs=build, chỉ `main`) — build native arm64, push lên `ghcr.io/<repo>/recsys-api:latest` + tag commit SHA, smoke test lại image đã push.

#### 3.5.2. `train.yml` — Weekly training

Trigger: `schedule: 0 2 * * 0` + `workflow_dispatch` (input: `model_type`, `force_preprocess`).

Chạy trên **self-hosted** (Mac). Steps:

1. **Install deps** — cài `requirements.txt`. Trên macOS: `brew install libomp` + pip install LightFM từ git master với `CFLAGS="-Xpreprocessor -fopenmp -I$LIBOMP/include"` để **bật OpenMP** (nếu không thì pip wheel chạy single-thread → chậm 4–8×).
2. **Set persistent paths** — `$GITHUB_WORKSPACE/data/processed`, `$GITHUB_WORKSPACE/artifacts` để khớp bind-mount của `api` container.
3. **Preprocess** — skip nếu đã có `train.parquet` và `force_preprocess=false`.
4. **Train** — `python -m src.training.train --model lightfm`, verify `model.pkl` + `model_meta.json` tồn tại.
5. **Reload API** — `docker compose -f docker-compose.prod.yml restart api`, sau đó retry **24 lần × 5s** curl `/health`, dừng khi thấy `trained_at` field trong response.
6. **Report** — in `model_meta.json` ra summary.

#### 3.5.3. `deploy.yml` — Deploy

Trigger: `workflow_run` của `CI` xong xanh trên `main`, hoặc `workflow_dispatch` (input `image_tag`).

Chạy trên **self-hosted** (Mac), environment `production`. Steps:

1. Resolve `image_tag` (từ input hoặc `head_sha` của CI run).
2. Log in `ghcr.io`.
3. **Save previous image** → `docker tag recsys-api:deploy recsys-api:previous` (enable rollback).
4. Pull image → retag thành `recsys-api:deploy` (compose dùng biến `API_IMAGE`).
5. `docker compose pull --ignore-pull-failures` + `up -d --remove-orphans`.
6. **Health check loop** — 12 lần × 5s, phải trả **status 200**.
7. **Rollback on failure** — `docker tag recsys-api:previous recsys-api:deploy` rồi `down` + `up -d` lại.

---

## 4. Function / class reference

Bảng tra cứu nhanh — mỗi hàng = một symbol then chốt.

| File | Symbol | Trách nhiệm (1 câu) |
|---|---|---|
| [`src/config.py`](../src/config.py) | `Settings` | Pydantic-settings singleton, env prefix `RECSYS_`, chứa path + hyperparam + MLflow URI. |
| [`src/data/preprocessing.py`](../src/data/preprocessing.py) | `preprocess()` | Orchestrator: load → filter cold-start → encode ID → temporal split → save parquet. |
| " | `filter_cold_start()` | Lặp cho đến khi hội tụ: loại user < 20 rating, movie < 10 rating. |
| " | `temporal_split()` | Sort theo timestamp, chia 70/15/15. |
| [`src/features/lightfm_dataset.py`](../src/features/lightfm_dataset.py) | `build_lightfm_dataset()` | Đọc CSV thô, random split 80/20, `Dataset.fit` users+items+genres, build interactions + item_features, trả `LightFMBundle`. |
| " | `_parse_genres()` | Tách `"Action\|Comedy"` thành list, loại `"(no genres listed)"`. |
| [`src/models/baseline.py`](../src/models/baseline.py) | `BaseRecommender` | ABC với `fit()` + `recommend()`. |
| " | `RandomRecommender` | Baseline: random item chưa xem, score uniform. |
| " | `PopularityRecommender` | Baseline: sort item theo số interaction, score = pop/max_pop. |
| [`src/models/lightfm_hybrid.py`](../src/models/lightfm_hybrid.py) | `LightFMRecommender.fit()` | Train LightFM WARP, cache `_user_seen` từ CSR indptr/indices, cache `_item_embeddings` từ `get_item_representations`. |
| " | `LightFMRecommender.recommend()` | `model.predict` → set seen=-inf → argpartition top-n. |
| " | `LightFMRecommender.similar_items()` | Cosine similarity trên embeddings, loại chính nó. |
| [`src/training/train.py`](../src/training/train.py) | `run_training_pipeline()` | `mlflow.start_run` → train_and_evaluate → log_metrics → dump artifact + meta. |
| " | `_evaluate_lightfm()` | Gọi `precision_at_k / recall_at_k / auc_score` có truyền `train_interactions` để chống leakage. |
| " | `_build_artifact()` | Đóng gói dict {model, item_features, embeddings, mapping, user_seen, movies, metrics}. |
| [`src/serving/app.py`](../src/serving/app.py) | `lifespan()` | Load artifact + config logging khi startup. |
| " | `CorrelationIdMiddleware` | Gắn `X-Request-ID` + emit log `request_completed` JSON. |
| [`src/serving/state.py`](../src/serving/state.py) | `service`, `start_time` | Process-wide singletons dùng chung giữa các router. |
| [`src/serving/recommender.py`](../src/serving/recommender.py) | `RecommenderService.load()` | `joblib.load` → check `model_type=="lightfm"` → unpack → build `_movie_meta` → tính `_popular_items` → load `model_meta.json`. |
| " | `RecommenderService.recommend()` | LRU cache 1024 entries → cold-start fallback → `model.predict` → exclude seen → top-k → sigmoid. |
| " | `RecommenderService._format_item()` | Wrap `{movie_id, title, score=sigmoid(raw), genres}`. |
| " | `RecommenderService._compute_popular_items()` | Đếm item xuất hiện trong `user_seen`, trả top-100 có `score=count/max_count`. |
| " | `RecommenderService.similar_items()` | Cosine similarity, raise `UnknownMovieError` nếu movie không có trong catalog. |
| [`src/serving/metrics.py`](../src/serving/metrics.py) | `REQUEST_COUNTER`, `COLD_START_COUNTER`, `LATENCY_HISTOGRAM` | 3 Prometheus metric canonical, định nghĩa module-level để không duplicate khi test re-import. |
| [`src/serving/logging_config.py`](../src/serving/logging_config.py) | `RecsysJsonFormatter`, `configure_logging()` | JSON formatter thêm `service`/`service_version`/`env`/`timestamp`; bắt lại các logger uvicorn/fastapi. |
| [`src/serving/routers/recommend.py`](../src/serving/routers/recommend.py) | `recommend()`, `recommend_batch()` | POST `/recommend` + `/recommend/batch`, đếm cold-start, observe latency, map Exception → 500/503. |
| [`src/serving/routers/similar.py`](../src/serving/routers/similar.py) | `similar_movies()` | GET `/similar/{movie_id}` → 404 nếu không có. |
| [`src/serving/routers/health.py`](../src/serving/routers/health.py) | `health()`, `_format_uptime()` | GET `/health` trả uptime format + model_meta. |
| [`src/serving/routers/prometheus.py`](../src/serving/routers/prometheus.py) | `metrics()` | GET `/metrics` trả `generate_latest()`. |
| [`src/serving/schemas.py`](../src/serving/schemas.py) | `RecommendRequest/Response`, `HealthResponse`, … | Pydantic v2 schemas cho request/response. |
| [`monitoring/airflow/dags/retrain_dag.py`](../monitoring/airflow/dags/retrain_dag.py) | `check_processed_data()` | BranchPython: nhảy sang `preprocess_data` hay `train_model`. |
| " | `reload_api()` | PythonOperator: `docker.from_env().containers.list(filters={"label":"com.docker.compose.service=api"}).restart()`. |
| [`monitoring/evidently/app.py`](../monitoring/evidently/app.py) | `_compute_drift_report()` | Load reference (sample 10k từ `train.parquet`) + current → `Report(DataDriftPreset()).run()` → set `DRIFT_SCORE`, `DRIFT_DETECTED` gauge. |
| " | `_refresh_drift_metrics_async()` | APScheduler interval 300s, chạy drift compute trên worker thread. |

---

## 5. Monitoring — như thế nào?

Hệ thống có **4 lớp quan sát** + 1 lớp experiment tracking. Tất cả plumbing đều có trong `docker-compose.prod.yml`.

### 5.1. Application metrics (Prometheus)

FastAPI expose `/metrics` qua `prometheus_client`. 3 metric canonical (định nghĩa ở [`src/serving/metrics.py`](../src/serving/metrics.py)):

| Metric | Type | Labels | Ý nghĩa |
|---|---|---|---|
| `recommendation_requests_total` | Counter | `status` (success/error/not_found), `endpoint` | Đếm request theo endpoint + trạng thái. |
| `cold_start_fallback_total` | Counter | — | Đếm request có user_id không có trong training map → fallback popular. |
| `recommendation_latency_seconds` | Histogram | `endpoint` | Phân phối latency end-to-end. |

Scrape config ([`monitoring/prometheus.yml`](../monitoring/prometheus.yml)):

```yaml
- api:8000              scrape 10s   metrics_path /metrics
- evidently:8001        scrape 60s
- mlflow:5000           scrape 30s
- cadvisor:8080         scrape 15s
- node-exporter:9100    scrape 15s
```

Retention: 30 ngày (`--storage.tsdb.retention.time=30d`).

### 5.2. Data drift (Evidently)

Service `evidently:8001` ([`monitoring/evidently/app.py`](../monitoring/evidently/app.py)):

- **APScheduler** chạy `_refresh_drift_metrics` mỗi **300 giây** (`EVIDENTLY_REFRESH_INTERVAL_SECONDS`).
- So sánh `reference.parquet` (sample 10 000 dòng từ `train.parquet`, seed=42, cache lại) với `current.parquet` dùng **`evidently.metric_preset.DataDriftPreset`**.
- Chỉ so sánh các cột chung giữa reference và current.
- Update Prometheus gauges:
  - `evidently_drift_score` — `dataset_drift_share` (tỉ lệ feature bị drift).
  - `evidently_drift_detected` — 1 nếu `dataset_drift` là true.
  - `evidently_reports_total` — counter số lần tính report.
- Endpoint phụ: `GET /health`, `GET /status`, `GET /report` (HTML Evidently), `GET /metrics`.

Hiện tại chưa có pipeline ghi `current.parquet` từ production log → Evidently dùng fake sample để demo. Muốn thật thì cần job đọc Elasticsearch `recsys-logs-*` sink ra parquet định kỳ.

### 5.3. Container metrics (cAdvisor) + Host metrics (node-exporter)

- `cadvisor` mount `/` + `/sys` + `/var/lib/docker` read-only, expose `container_memory_usage_bytes`, `container_cpu_usage_seconds_total`, …
- `node-exporter` mount `/:/host:ro`, `pid: host`. **Caveat trên macOS**: chỉ thấy VM Docker Desktop, không phải host Mac thật.

### 5.4. Alert rules

File: [`monitoring/prometheus-rules.yml`](../monitoring/prometheus-rules.yml). Hai group:

**Group `system`** (evaluation mỗi 30s):

- `HighContainerMemory` — container_memory / limit > 0.8 trong 5m → warning.
- `HighContainerCPU` — rate(container_cpu) × 100 > 85 trong 5m → warning.
- `HighHostCPU` — 100 − avg(rate(idle)) × 100 > 85 trong 5m → warning.
- `HighHostMemory` — (1 − avail/total) × 100 > 85 trong 5m → warning.
- `DiskSpaceLow` — filesystem{mountpoint="/host"} > 85% trong 10m → warning.

**Group `recsys`** (evaluation mỗi 30s):

- `RecsysModelColdStart` — `increase(recsys_cold_start_total[5m]) > 10` trong 2m → critical (có thể model chưa load).
- `RecsysHighErrorRate` — tỉ lệ 5xx > 5% trong 5m → critical.
- `EvidentlyDriftDetected` — `evidently_drift_detected == 1` trong 10m → warning.

**Alertmanager chưa được wired** — chỉ xem trạng thái alert ở `http://localhost:9090/alerts`.

### 5.5. Grafana dashboards

Auto-provisioned qua [`monitoring/grafana/provisioning/`](../monitoring/grafana/provisioning) + volume mount dashboard JSON. 2 dashboard:

- [`monitoring/grafana/dashboard.json`](../monitoring/grafana/dashboard.json) — API metrics + drift.
- [`monitoring/grafana/system-dashboard.json`](../monitoring/grafana/system-dashboard.json) — container + host.

Plugin: `grafana-piechart-panel`. Credentials: `admin` / `$GRAFANA_PASSWORD` (bắt buộc set trong `.env`).

### 5.6. Logging — ELK pipeline

**Nguồn log:** API emit JSON structured ra stdout. Formatter [`RecsysJsonFormatter`](../src/serving/logging_config.py):

```json
{
  "timestamp": "2026-04-19T10:00:00Z",
  "level": "INFO",
  "logger": "src.serving.routers.recommend",
  "message": "recommendation_served",
  "service": "recsys-api",
  "service_version": "lightfm_v1",
  "env": "production",
  "request_id": "...",
  "user_id": 1,
  "movie_ids": [318, 2571, ...],
  "latency_ms": 12.345,
  "cold_start": false,
  "event": "recommendation_served",
  "status": "success"
}
```

**Đường đi:**

```
FastAPI stdout (JSON)
  → /var/lib/docker/containers/*/*.log
  → Filebeat (type: container, add_docker_metadata)
  → processors:
      - drop_event nếu container không phải {api, mlflow, airflow, evidently}
      - decode_json_fields(message) → parse JSON vào field top-level
      - drop_fields [timestamp]              # né ES date mapping 400
      - rename event → log_event              # né ECS object conflict
               service → service_name         # (ECS có service là object)
               user_agent → http_user_agent   # (ECS có user_agent là object)
      - drop filebeat heartbeat noise
  → Elasticsearch index "recsys-logs-YYYY.MM.dd"
  → Kibana (kibana-init auto-load saved objects từ saved-objects.ndjson)
```

Template index: 1 shard, 0 replica, ILM disabled (dữ liệu demo, không xoay vòng).

### 5.7. Experiment tracking — MLflow

- **Backend store:** Postgres DB `mlflow` (user/pass từ `POSTGRES_PASSWORD`).
- **Artifact store:** MinIO bucket `mlflow-artifacts` qua S3 API (`s3://mlflow-artifacts/`).
- **Mode `--serve-artifacts`**: MLflow server làm proxy upload/download artifact → runner training không cần S3 credentials, chỉ cần HTTP tới MLflow.
- **Mỗi run log:**
  - `params`: no_components, loss, lr, epochs, num_threads, test_size, split_seed
  - `metrics`: precision_at_k, recall_at_k, auc, f1_score, n_users_evaluated, training_time_seconds
  - `artifacts`: `model.pkl`, `model_meta.json`
  - `tags`: model_type, dataset_version=`ml-latest-small`, split_strategy=`random_80_20`

UI: `http://localhost:5000` (localhost bind, cần SSH tunnel).

---

## 6. Invariant train/serve contract

Đây là **quy tắc cốt lõi** của hệ thống — nếu phá là service không load được model.

### 6.1. Host dir `$RECSYS_ARTIFACTS_DIR` là single source of truth

Dir này bind-mount vào:

- `/app/artifacts` (container `api`) — nơi API đọc `model.pkl` + `model_meta.json` khi startup.
- `/opt/airflow/artifacts` (mọi container airflow) — nơi `train_model` task ghi ra.
- `$GITHUB_WORKSPACE/artifacts` (self-hosted runner train.yml) — nơi GitHub Actions ghi.

Default: `./artifacts` (path tương đối với `docker-compose.prod.yml`).

### 6.2. Training chỉ cần ghi đủ 2 file

```
$RECSYS_ARTIFACTS_DIR/
├── model.pkl           # joblib dump artifact dict
└── model_meta.json     # trained_at, mlflow_run_id, git_sha, metrics
```

### 6.3. Reload = restart API container

API **chỉ load model trong `lifespan`** (startup). Muốn lấy model mới:

```bash
docker compose -f docker-compose.prod.yml restart api
```

Không dùng `docker cp`. Không SIGHUP. Cả Airflow DAG và `train.yml` đều restart container.

### 6.4. Artifact phải có `model_type == "lightfm"`

Nếu `joblib.load(model.pkl)` không phải LightFM, `RecommenderService.load` raise `RecommenderServiceError` → `/health` status `unavailable` → `/recommend` trả 503. Đây là safety check: nếu lỡ retrain baseline (random/popular) thì serving sẽ từ chối load.

---

## 7. Cheatsheet vận hành

Tổng hợp từ [`Makefile`](../Makefile):

```bash
# Setup
make setup                      # pip install deps dev + test
make gen-secrets                # sinh .env với password ngẫu nhiên

# Data + training local
make data                       # python -m src.data.preprocessing
make train                      # train LightFM, log MLflow
make train-all                  # train random + popular + lightfm

# Stack Docker
make docker-up                  # compose up -d --build (full stack)
make docker-down                # compose down
make docker-logs SERVICE=api    # tail logs 1 service
make docker-ps                  # ps services

# Operator
make health                     # curl /health trên tất cả service
make train-and-reload           # train rồi restart api
make mlflow-ui                  # open http://localhost:5000

# Quality
make test                       # pytest + coverage
make test-quick                 # stop on first failure
make lint                       # ruff check --fix + ruff format
make lint-check                 # CI mode
make typecheck                  # mypy (warning-only)
```

URL khi stack lên (localhost):

| UI | URL | Credentials |
|---|---|---|
| API docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / `$GRAFANA_PASSWORD` |
| Kibana | http://localhost:5601 | — |
| Airflow | http://localhost:8080 | admin / `$AIRFLOW_ADMIN_PASSWORD` |
| MinIO Console | http://localhost:9001 | `$MINIO_ROOT_USER` / `$MINIO_PASSWORD` |
| Prometheus | http://localhost:9090 | — |
| Evidently | http://localhost:8001 | — |

Remote server: SSH tunnel trước (`ssh -L 5000:localhost:5000 -L 3000:localhost:3000 …`).

---

## 8. Quick reference paths

| Chủ đề | File |
|---|---|
| Config singleton | [`src/config.py`](../src/config.py) |
| Preprocessing | [`src/data/preprocessing.py`](../src/data/preprocessing.py) |
| LightFM dataset builder | [`src/features/lightfm_dataset.py`](../src/features/lightfm_dataset.py) |
| LightFM model wrapper | [`src/models/lightfm_hybrid.py`](../src/models/lightfm_hybrid.py) |
| Baselines | [`src/models/baseline.py`](../src/models/baseline.py) |
| Training pipeline | [`src/training/train.py`](../src/training/train.py) |
| FastAPI app + middleware | [`src/serving/app.py`](../src/serving/app.py) |
| Recommender service | [`src/serving/recommender.py`](../src/serving/recommender.py) |
| Prometheus metrics | [`src/serving/metrics.py`](../src/serving/metrics.py) |
| JSON logging | [`src/serving/logging_config.py`](../src/serving/logging_config.py) |
| Routers | [`src/serving/routers/`](../src/serving/routers/) |
| Pydantic schemas | [`src/serving/schemas.py`](../src/serving/schemas.py) |
| Airflow DAG | [`monitoring/airflow/dags/retrain_dag.py`](../monitoring/airflow/dags/retrain_dag.py) |
| Evidently service | [`monitoring/evidently/app.py`](../monitoring/evidently/app.py) |
| Prometheus config | [`monitoring/prometheus.yml`](../monitoring/prometheus.yml) |
| Alert rules | [`monitoring/prometheus-rules.yml`](../monitoring/prometheus-rules.yml) |
| Filebeat config | [`monitoring/filebeat/filebeat.yml`](../monitoring/filebeat/filebeat.yml) |
| Grafana dashboards | [`monitoring/grafana/`](../monitoring/grafana) |
| Kibana init | [`monitoring/kibana/`](../monitoring/kibana) |
| Docker compose | [`docker-compose.prod.yml`](../docker-compose.prod.yml) |
| API Dockerfile | [`Dockerfile`](../Dockerfile) |
| CI | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) |
| Train | [`.github/workflows/train.yml`](../.github/workflows/train.yml) |
| Deploy | [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) |
| Makefile | [`Makefile`](../Makefile) |

---

**Kết luận**: toàn bộ hệ thống xoay quanh một **contract duy nhất**: `$RECSYS_ARTIFACTS_DIR` chứa `model.pkl` + `model_meta.json`. Ai viết vào đó (local `make train`, Airflow DAG, hay GitHub Actions `train.yml`) đều restart container `api` xong là deploy xong. Monitoring tách rời thành 3 bộ (metrics qua Prometheus, drift qua Evidently, log qua ELK) nhưng đều consume từ cùng một nguồn: stdout JSON + `/metrics` endpoint của API. Khi thay model hay hyperparam, chỉ cần sửa `src/config.py` + rerun training — serving tự reload.
