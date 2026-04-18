# Architecture

One-page tour of how data flows through the system and where each component lives.

## High-level diagram

```
┌───────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  MovieLens CSV    │───▶ │  Preprocessing       │───▶ │  data/processed/    │
│  (raw)            │     │  (src/data/*.py)     │     │  train.parquet …    │
└───────────────────┘     └──────────────────────┘     └─────────┬───────────┘
                                                                 │
                                                                 ▼
┌──────────────────────┐                              ┌────────────────────┐
│  MLflow + MinIO      │◀─── log params / metrics ───│  Training          │
│  (tracks experiments)│                              │  (src/training)    │
└──────────────────────┘                              └─────────┬──────────┘
                                                                 │ writes
                                                                 ▼
                                                      ┌────────────────────┐
                                                      │ artifacts/         │
                                                      │  model.pkl         │
                                                      │  model_meta.json   │
                                                      └─────────┬──────────┘
                                                                 │ bind mount
                                                                 ▼
                                                      ┌────────────────────┐
                                                      │ FastAPI container  │
                                                      │ /app/artifacts/    │
                                                      │ loads at startup   │
                                                      └─────────┬──────────┘
                                                                 │ HTTP
                                                                 ▼
                                             ┌────────────────────────────────┐
                                             │ Client ← JSON recommendations  │
                                             └────────────────────────────────┘
```

## Critical path: train → serve

This is the **most important** thing to understand about the deployment.

1. Training writes **both** `model.pkl` and `model_meta.json` into `$RECSYS_ARTIFACTS_DIR`
   (host dir: `./artifacts` by default; on the runner: `$HOME/recsys-artifacts`).
2. The same host dir is **bind-mounted** into the API container at `/app/artifacts`.
   There is no `docker cp` step — the file is already visible.
3. A restart (`docker compose restart api`) forces the FastAPI lifespan to re-load
   `model.pkl` from disk.
4. `GET /health` returns the new `trained_at` / `mlflow_run_id` so operators can
   verify the reload landed.

**Why a bind mount (not a named volume):** the training job can live inside Airflow
(in the network) or on the host (GitHub Actions runner) — both need to write to the
same path that the API reads from. A shared host directory is the one invariant
that works for either training location.

## Layout by concern

```
src/
  config.py           pydantic-settings — the source of truth for paths + env vars
  data/               ingestion + preprocessing
  features/           user/item/interaction feature engineering
  models/             baseline, ALS, content-based recommenders
  evaluation/         precision/recall/NDCG/coverage metrics
  training/train.py   pipeline: load → train → evaluate → log → save
  serving/
    app.py            FastAPI wiring: middleware, lifespan, include_router
    state.py          shared singletons (service instance, uptime)
    metrics.py        Prometheus Counters/Histograms (one canonical copy)
    recommender.py    RecommenderService — loads model.pkl and serves predictions
    schemas.py        pydantic request/response models
    routers/
      health.py       GET /health (+ model metadata)
      recommend.py    POST /recommend, POST /recommend/batch
      similar.py      GET /similar/{movie_id}
      prometheus.py   GET /metrics
    logging_config.py  JSON log formatter for Elasticsearch/Kibana
```

## Process topology (production compose stack)

```
           ┌─────────────┐
           │  postgres   │── MLflow backend DB + Airflow metadata DB
           └──────┬──────┘
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌──────────┐
│ mlflow  │  │ airflow │  │ airflow  │
│ server  │  │ web     │  │ scheduler│
└────┬────┘  └─────────┘  └────┬─────┘
     │                          │
     │           ┌──────────────┘
     │           │ shared bind-mount: ${RECSYS_ARTIFACTS_DIR}
     ▼           ▼
┌─────────────────────┐        ┌──────────────┐
│   api (FastAPI)     │◀───────│ evidently    │─── drift reports
└──────────┬──────────┘        └──────────────┘
           │ /metrics, /logs
           ▼
┌──────────────┐  ┌──────────┐  ┌───────────┐
│ prometheus   │  │ filebeat │──│elasticsearch│──┐
└──────┬───────┘  └──────────┘  └───────────┘   │
       │                                          │
       ▼                                          ▼
┌──────────────┐                          ┌──────────────┐
│  grafana     │                          │   kibana     │
└──────────────┘                          └──────────────┘

MinIO (S3): artifact store for MLflow.
```

## Environment variables (the ones that matter)

| Variable | Role |
|---|---|
| `RECSYS_DATA_DIR` | Host dir bind-mounted as `/app/data`, `/opt/airflow/data`. Default `./data`. |
| `RECSYS_ARTIFACTS_DIR` | Host dir bind-mounted as `/app/artifacts`, `/opt/airflow/artifacts`. Default `./artifacts`. |
| `RECSYS_MODEL_PATH` | Absolute path inside the container. Default `/app/artifacts/model.pkl`. |
| `RECSYS_MLFLOW_TRACKING_URI` | `http://mlflow:5000` inside the network, `http://localhost:5000` from the host. |
| `POSTGRES_PASSWORD`, `MINIO_PASSWORD`, `GRAFANA_PASSWORD`, `AIRFLOW_*` | Secrets. Generated by `scripts/gen-secrets.sh`. |

## Further reading

- Operational runbook: [OPERATIONS.md](OPERATIONS.md)
- Local dev without Docker: [DEVELOPMENT.md](DEVELOPMENT.md)
- Monitoring stack deep-dive: [monitoring/README.md](monitoring/README.md)
