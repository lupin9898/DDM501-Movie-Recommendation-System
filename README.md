# Movie Recommendation System

End-to-end MovieLens recommendation pipeline: ingest → preprocess → train (ALS,
content-based, baselines) → FastAPI service → MLOps stack (MLflow, Airflow,
Prometheus, Grafana, Evidently).

## Quick Start

```bash
git clone <repo-url>
cd DDM501-Movie-Recommendation-System

make gen-secrets          # writes .env with random passwords
make docker-up            # boots the full stack
make health               # curl every service's /health
```

After training (via `make train-and-reload` or the `train.yml` workflow), hit
`GET /health` — the `trained_at` and `model_type` fields confirm which model
the API is currently serving.

```bash
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "top_k": 5}'
```

## Service URLs

| Service | URL | Login |
|---|---|---|
| API | http://localhost:8000 | — |
| API Docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `$GRAFANA_PASSWORD` |
| Airflow | http://localhost:8080 | `admin` / `$AIRFLOW_ADMIN_PASSWORD` |
| Kibana | http://localhost:5601 | — |
| Prometheus | http://localhost:9090 | — |
| MinIO Console | http://localhost:9001 | `$MINIO_ROOT_USER` / `$MINIO_PASSWORD` |
| Evidently | http://localhost:8001 | — |

## Tech Stack

| Layer | Tech |
|---|---|
| Language | Python 3.11 |
| ML | `implicit` (ALS), scikit-learn, SciPy |
| API | FastAPI + Uvicorn |
| Experiment tracking | MLflow + PostgreSQL + MinIO |
| Orchestration | Apache Airflow (LocalExecutor) |
| Drift detection | Evidently |
| Metrics | Prometheus + Grafana |
| Log shipping | Filebeat → Elasticsearch → Kibana |
| Container | Docker Compose |
| CI/CD | GitHub Actions (self-hosted runner, Mac arm64) |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit together, data flow, where files live.
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — operator runbook: boot, verify, diagnose incidents.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — local dev without Docker, testing conventions.
- [docs/monitoring/](docs/monitoring/) — monitoring stack setup and troubleshooting.
- [RUNNER_SETUP.md](RUNNER_SETUP.md) — self-hosted GitHub Actions runner setup.

## Benchmark (ml-latest-small)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Random | ~0.01 | ~0.005 | ~0.01 |
| Popularity | ~0.05 | ~0.03 | ~0.04 |
| ALS (factors=100) | ~0.10–0.15 | ~0.06–0.10 | ~0.10–0.15 |

## Makefile cheat sheet

```bash
make help                     # list every target
make gen-secrets              # write .env with random passwords
make docker-up / docker-down  # start / stop full stack
make docker-logs SERVICE=api  # tail a service's logs
make health                   # check every service's /health
make train-and-reload         # train locally then restart API
make lint / test / typecheck  # quality gates
```

## CI/CD workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push main/develop, PR | lint → test → docker build → smoke test → push ghcr.io |
| `deploy.yml` | after CI green on main / manual | pull image → compose up → health check → auto-rollback |
| `train.yml` | weekly cron / manual | download → preprocess → train → restart API → verify `/health` |
