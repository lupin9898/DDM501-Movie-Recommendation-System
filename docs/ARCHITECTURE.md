# Architecture

One-page tour of how data flows through the system, which components own what,
and why each piece is wired the way it is. Pair this with
[OPERATIONS.md](OPERATIONS.md) (runbook) and [DEVELOPMENT.md](DEVELOPMENT.md)
(local dev).

## High-level data flow

```
┌───────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  MovieLens ZIP    │───▶ │  Preprocessing       │───▶ │  $RECSYS_DATA_DIR/  │
│  (raw)            │     │  (src/data/*.py)     │     │    processed/*.pq   │
└───────────────────┘     └──────────────────────┘     └─────────┬───────────┘
                                                                 │
                                                                 ▼
┌──────────────────────┐                              ┌────────────────────┐
│  MLflow + MinIO      │◀─── log params/metrics ─────│  Training          │
│  (experiment store)  │                              │  (src/training)    │
└──────────────────────┘                              └─────────┬──────────┘
                                                                 │ writes
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │ $RECSYS_ARTIFACTS_DIR│
                                                      │   model.pkl          │
                                                      │   model_meta.json    │
                                                      └─────────┬────────────┘
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

                     Evidently compares train set ⇄ live request logs
                     → drift metrics → Prometheus → Grafana panel
```

## Critical path: train → serve

This is the **most important** invariant in the deployment.

1. Training writes **both** `model.pkl` and `model_meta.json` into
   `$RECSYS_ARTIFACTS_DIR`. On the self-hosted runner this resolves to
   `$HOME/recsys-artifacts`; locally it defaults to `./artifacts`.
2. The same host dir is **bind-mounted** into the API container at
   `/app/artifacts`. There is no `docker cp` — the file is already visible.
3. `docker compose restart api` forces the FastAPI lifespan to re-load
   `model.pkl` from disk.
4. `GET /health` returns the fresh `trained_at` / `mlflow_run_id` / `git_sha`
   so operators can verify the reload landed.

**Why a bind mount (not a named volume):** training can run inside Airflow
(in-network) or on the GitHub Actions runner (on the host). Both paths must
write to the same directory the API reads from. A shared host directory is the
only invariant that works for either training location.

## Source layout

```
src/
  config.py              pydantic-settings — single source of truth for paths and env vars
  data/                  MovieLens ingestion + preprocessing into train/val/test parquet
  features/              interaction matrix + user/item feature engineering
  models/                baseline (random, popularity), ALS, content-based
  evaluation/metrics.py  precision, recall, ndcg, f1, hit_rate, mrr, coverage
  training/train.py      load → build interactions → fit → evaluate → log → save
  serving/
    app.py               FastAPI wiring: middleware, lifespan, include_router
    state.py             shared singletons (service instance, uptime)
    metrics.py           Prometheus Counters/Histograms (one canonical copy)
    recommender.py       RecommenderService — loads model.pkl, serves predictions
    schemas.py           pydantic request/response models (incl. HealthResponse)
    logging_config.py    JSON log formatter for Filebeat → Elasticsearch
    routers/
      health.py          GET /health (status, model_version, trained_at, git_sha)
      recommend.py       POST /recommend, POST /recommend/batch
      similar.py         GET /similar/{movie_id}
      prometheus.py      GET /metrics
```

## Process topology (compose stack)

```
              ┌──────────────┐
              │   postgres   │  MLflow backend DB + Airflow metadata DB
              └──────┬───────┘
                     │
       ┌─────────────┼────────────────┐
       ▼             ▼                ▼
  ┌─────────┐   ┌─────────┐   ┌─────────────┐
  │ mlflow  │   │ airflow │   │   airflow   │
  │ server  │   │   web   │   │  scheduler  │
  └────┬────┘   └─────────┘   └──────┬──────┘
       │  artifacts:                 │
       ▼  S3 to MinIO        shared bind-mount: $RECSYS_ARTIFACTS_DIR
  ┌─────────┐                        │
  │  minio  │                        ▼
  └─────────┘           ┌─────────────────────┐
                        │   api (FastAPI)     │◀──┐
                        └──────────┬──────────┘   │ /recsys-data:ro
                                   │              │
                                   │ /metrics  ┌──┴───────────┐
                                   ▼            │  evidently   │── drift reports
                         ┌────────────────┐    └──────────────┘
                         │  prometheus    │◀── scrape(api, evidently,
                         │  + rules.yml   │            cadvisor, node-exporter)
                         └──────┬─────────┘
                                │                 ┌──────────────┐
                                ▼                 │  cadvisor    │ per-container
                         ┌────────────┐           │  node-exporter│ host (VM)
                         │  grafana   │           └──────────────┘
                         │ (2 boards) │
                         └────────────┘

  Logging: api → stdout (JSON) → filebeat → elasticsearch → kibana
```

All services share the `backend` network. Ports are bound to `127.0.0.1` only
(except the runner's reverse proxy, if any) to keep the stack off the public
internet.

## Evaluation metrics

`src/evaluation/metrics.py` exposes the full set; `evaluate_model()` returns
all of them so MLflow and `model_meta.json` record everything:

| Metric | Range | What it tells you |
|---|---|---|
| `precision_at_k` | 0–1 | Of the top-K recommended items, how many are actually liked. |
| `recall_at_k` | 0–1 | Of the items the user likes, how many appear in top-K. |
| `ndcg_at_k` | 0–1 | Same as precision but position-weighted (higher rank counts more). |
| **`f1_score`** | 0–1 | Harmonic mean of precision and recall — single "balance" number. |
| **`hit_rate_at_k`** | 0–1 | Fraction of users with at least one relevant item in top-K. |
| **`mrr`** | 0–1 | Mean reciprocal rank — `1/position` of the first relevant hit. |
| `coverage` | 0–1 | Fraction of the catalog that appears in any user's top-K list. |

Rule of thumb on ml-latest-small: popularity ≈ 0.03 P@10, tuned ALS ≈ 0.05
P@10. Recsys metrics are much smaller than classification metrics — this is
normal, not a bug.

## ML techniques in the ALS path

Two non-obvious choices in `src/models/collaborative.py`:

1. **BM25 weighting** (default `use_bm25=True`). Before training, the
   interaction matrix is passed through `bm25_weight(K1=100, B=0.8)` from
   `implicit.nearest_neighbours`. This down-weights popular items so the model
   learns tail/niche factors instead of collapsing onto blockbusters.
   Raised precision@10 from 0.039 → 0.054 on the small dataset.
2. **Confidence-weighted loss.** The (possibly BM25-weighted) matrix is
   multiplied by `alpha` (default 100) before fit. ALS then treats larger
   values as higher-confidence positives — standard practice for implicit
   feedback as introduced in Hu–Koren–Volinsky (2008).

`exclude_seen=True` in `recommend()` masks items the user already rated, so the
service never recommends something they've already watched.

## Observability

Three layers, all scraped by Prometheus:

| Layer | Source | URL | What it measures |
|---|---|---|---|
| Application | FastAPI middleware | `http://localhost:8000/metrics` | request rate, latency p50/p95/p99, cold-start fallbacks |
| Drift | Evidently service | `http://localhost:8001/metrics` | `evidently_drift_score`, `evidently_drift_detected` |
| Container | cAdvisor | `http://localhost:8081` | per-container CPU, memory, network, disk |
| Host | node-exporter | `http://localhost:9100/metrics` | host (Docker Desktop VM) CPU, RAM, disk, load |

Two Grafana dashboards auto-provisioned:

- **Recommendation System Dashboard** — API + drift.
- **System & Container Metrics** — host + container.

Alert rules live in `monitoring/prometheus-rules.yml` (8 rules: container/host
resource pressure, cold-start spikes, 5xx rate, drift detection). View at
`http://localhost:9090/alerts`. Alertmanager is not wired yet — fires show in
the Prometheus UI only.

**macOS caveat:** node-exporter inside Docker Desktop only sees the Linux VM,
not macOS itself. For native host metrics run `brew install node_exporter`.

## Environment variables (the ones that matter)

| Variable | Role |
|---|---|
| `RECSYS_DATA_DIR` | Host dir bind-mounted as `/app/data`, `/opt/airflow/data`, `/app/recsys-data:ro`. Default `./data`. |
| `RECSYS_ARTIFACTS_DIR` | Host dir bind-mounted as `/app/artifacts`, `/opt/airflow/artifacts`. Default `./artifacts`. **Must match the path training writes to** (see train.yml). |
| `RECSYS_MODEL_PATH` | Absolute path inside the API container. Default `/app/artifacts/model.pkl`. |
| `RECSYS_MLFLOW_TRACKING_URI` | `http://mlflow:5000` inside the network, `http://localhost:5000` from the host. |
| `POSTGRES_PASSWORD`, `MINIO_PASSWORD`, `GRAFANA_PASSWORD`, `AIRFLOW_*` | Secrets generated by `scripts/gen-secrets.sh`. |

## First files to read (for newcomers)

| File | Why |
|---|---|
| `src/config.py` | Every knob in the system — read this first. |
| `src/training/train.py` | The training pipeline end to end. |
| `src/serving/app.py` | FastAPI wiring; the routers it includes live in `src/serving/routers/`. |
| `docker-compose.prod.yml` | Full prod topology in one file. |
| `monitoring/prometheus.yml` + `prometheus-rules.yml` | Scrape targets and alerts. |
| `.github/workflows/{ci,deploy,train}.yml` | How CI, deploy, and scheduled training run. |

## Further reading

- Operator runbook: [OPERATIONS.md](OPERATIONS.md)
- Local dev without Docker: [DEVELOPMENT.md](DEVELOPMENT.md)
- Self-hosted runner setup: [../RUNNER_SETUP.md](../RUNNER_SETUP.md)
