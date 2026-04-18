# Operations Runbook

Everything an on-call operator needs to bring the stack up, verify it, and
diagnose the most common failure modes. See [ARCHITECTURE.md](ARCHITECTURE.md)
if you need to understand *why* something is wired the way it is.

## Cold start from scratch

```bash
# 1. Generate secrets (one-time)
make gen-secrets                  # writes .env with random passwords, mode 600

# 2. Boot the full stack
make docker-up                    # docker compose -f docker-compose.prod.yml up -d --build

# 3. Verify every service
make health                       # curls /health on API, MLflow, Grafana, Prometheus, Evidently
```

Expected HTTP status:

| Service | URL | Healthy response |
|---|---|---|
| API | http://localhost:8000/health | `{"status":"healthy", ...}` |
| MLflow | http://localhost:5000 | 200 |
| Grafana | http://localhost:3000 | 200 on `/api/health` |
| Prometheus | http://localhost:9090/-/healthy | 200 |
| Evidently | http://localhost:8001/health | 200 |
| Kibana | http://localhost:5601/api/status | 200 |
| Airflow | http://localhost:8080/health | 200 |
| MinIO | http://localhost:9000/minio/health/live | 200 |

## Training: how to verify the model reloaded

**Golden path (self-hosted runner):**

1. Trigger `train.yml` (GitHub → Actions → Train → Run workflow).
2. The workflow writes `model.pkl` + `model_meta.json` into the bind-mounted
   artifacts dir, then runs `docker compose restart api`.
3. Poll `/health`:
   ```bash
   curl -s http://localhost:8000/health | python -m json.tool
   ```
   Look at `trained_at` — it should be within the last minute.

**Local manual retrain:**

```bash
make train-and-reload
```

## Common incidents

### API returns 503 "Model not loaded"

- **Symptom:** `/recommend` returns 503; `/health` has `"status": "unavailable"`.
- **Root cause:** `model.pkl` is missing at `$RECSYS_ARTIFACTS_DIR/model.pkl`
  or corrupt, so the lifespan's `load()` raised.
- **Fix:**
  ```bash
  ls -la $RECSYS_ARTIFACTS_DIR           # confirm model.pkl + model_meta.json exist
  make docker-logs SERVICE=api           # look for "model_load_failed"
  make train-and-reload                  # re-train and restart
  ```
- **If still failing:** check that `$RECSYS_ARTIFACTS_DIR` is actually mounted
  into the container (`docker compose exec api ls /app/artifacts`).

### Postgres connection refused (MLflow, Airflow)

- **Symptom:** MLflow UI 502, Airflow scheduler restarting.
- **Root cause:** `postgres-init` didn't complete (bad password, DB migration
  failure) or `POSTGRES_PASSWORD` was rotated without restarting dependents.
- **Fix:**
  ```bash
  make docker-logs SERVICE=postgres
  make docker-logs SERVICE=postgres-init
  docker compose -f docker-compose.prod.yml up -d --force-recreate postgres-init
  docker compose -f docker-compose.prod.yml restart mlflow airflow-webserver airflow-scheduler
  ```

### MinIO bucket missing → MLflow can't log artifacts

- **Symptom:** training step fails with `Bucket 'mlflow-artifacts' not found`.
- **Fix:**
  ```bash
  docker compose -f docker-compose.prod.yml up -d --force-recreate minio-init
  ```

### GitHub Actions self-hosted runner: train fails before API reload

- **Symptom:** `train.yml` fails at "Reload API" step.
- **Possible causes:**
  - API container isn't running (start it first: `make docker-up`).
  - Host `$RECSYS_ARTIFACTS_DIR` and container `/app/artifacts` disagree —
    check `.env` matches what the workflow exports. Both **must** resolve to
    the same host directory.
- **Fix:**
  ```bash
  docker compose -f docker-compose.prod.yml exec api cat /app/artifacts/model_meta.json
  # If missing, the bind mount isn't wired. Verify .env on the runner.
  ```

### Filebeat drops logs / Kibana shows no data

- **Symptom:** `logs-recsys-*` index not populating.
- **Fix:**
  ```bash
  make docker-logs SERVICE=filebeat          # should show "Connection to ES established"
  curl -s localhost:9200/_cat/indices?v      # verify logs-recsys-* exists
  docker compose -f docker-compose.prod.yml up -d --force-recreate kibana-init
  ```

## Rotating secrets

```bash
./scripts/gen-secrets.sh --force          # overwrites .env with fresh values
docker compose -f docker-compose.prod.yml up -d --force-recreate postgres-init \
  mlflow airflow-init airflow-webserver airflow-scheduler grafana
```

**Warning:** rotating `POSTGRES_PASSWORD` after first boot requires updating it
inside Postgres too — `postgres-init` does this idempotently via `ALTER USER`.

## Rollback a bad deploy

`deploy.yml` tags the previous image before deploying. To revert manually:

```bash
docker tag recsys-api:previous recsys-api:local
docker compose -f docker-compose.prod.yml up -d --no-build api
curl -s localhost:8000/health
```

## Scaling down for dev

If you only need the API + MLflow (no Airflow/Elastic/Kibana), stop the heavy
services after boot:

```bash
docker compose -f docker-compose.prod.yml stop \
  airflow-webserver airflow-scheduler \
  elasticsearch kibana filebeat
```

## Observability — system + container metrics

Two exporters feed Prometheus with infrastructure metrics:

| Exporter | URL | What it measures |
|---|---|---|
| **cAdvisor** | http://localhost:8081 | Per-container CPU, memory, network, disk I/O |
| **node-exporter** | http://localhost:9100/metrics | Host (Docker Desktop Linux VM) CPU, memory, disk, network |

Grafana has two dashboards:

- **Recommendation System Dashboard** — API request rate, latency, error rate, model version, Evidently drift.
- **System & Container Metrics** — host CPU/RAM/disk/network + per-container CPU/RAM/network.

**macOS caveat:** node-exporter runs inside the Docker Desktop Linux VM, so its "host" metrics reflect the VM, not macOS itself. For native macOS metrics install via Homebrew: `brew install node_exporter`.

**Alert rules** (`monitoring/prometheus-rules.yml`) fire when:
- Container memory >80% of limit, CPU >85% (5 min)
- Host CPU >85%, memory >85% (5 min)
- Disk >85% (10 min)
- API cold-start spike, 5xx error rate >5%, drift detected (10 min)

View active alerts at http://localhost:9090/alerts. Wire Alertmanager later to actually notify.

## Where to look for more

- Application logs: `make docker-logs SERVICE=api` (or check Kibana → `logs-recsys-*`).
- API request metrics: Grafana → "Recommendation System Dashboard".
- System metrics: Grafana → "System & Container Metrics".
- Active alerts: http://localhost:9090/alerts.
- Training runs: MLflow UI at http://localhost:5000.
- Architecture / why things are wired this way: [ARCHITECTURE.md](ARCHITECTURE.md).
