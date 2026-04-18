# Development

Local workflow for writing code without the full Docker stack.

## Prerequisites

- Python 3.11 (strictly — `pyproject.toml` pins this)
- `uv` (preferred) or `pip`
- Docker (only if you want to run the API container locally)

## Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-test.txt
```

Or with `uv`:

```bash
uv venv --python 3.11
uv pip sync requirements.txt requirements-test.txt
```

## Config via .env

```bash
cp .env.example .env
# Edit only what you need — defaults are fine for local dev without Docker.
```

The API reads env vars with prefix `RECSYS_`, e.g. `RECSYS_MODEL_PATH`,
`RECSYS_LOG_LEVEL`. See [`src/config.py`](../src/config.py) for the full list.

## The inner dev loop

```bash
make lint                     # ruff check --fix + ruff format
make test                     # pytest with coverage
make test-quick               # pytest -x -q (stop on first failure)
make typecheck                # mypy src/  (warning-only for now)
```

Suggested sequence when adding a feature:

1. Write the test first → `pytest tests/unit/test_<module>.py::TestFoo -x`.
2. Implement the smallest thing that makes it pass.
3. `make lint && make test-quick`.
4. If you touched the API surface, also run the integration tests:
   `pytest tests/integration/ -x`.

## Running the API without Docker

```bash
# 1. Make sure you have data + a model
make data
make train                    # writes artifacts/model.pkl + artifacts/model_meta.json

# 2. Start the dev server (auto-reload)
make serve                    # uvicorn src.serving.app:app --reload --port 8000
```

Hit http://localhost:8000/docs for the Swagger UI.

## Running against the containerised API

When you want to test against the full stack but develop on the host:

```bash
make docker-up                # full stack up
make train-and-reload         # train locally, then restart the container
curl -s http://localhost:8000/health | python -m json.tool
```

## Tests — where things live

```
tests/
  unit/            pure-logic tests (mock I/O at the repo boundary)
    test_data.py
    test_model.py
  integration/     tests that exercise more than one layer
    test_api.py    FastAPI TestClient against a real RecommenderService
```

Conventions:

- **Mock at the boundary.** Unit tests that need data should stub `pd.read_parquet`
  at the repo boundary, not patch service internals.
- **One happy path + one edge case** per new function, minimum.
- **Test file mirrors source:** `src/services/graph.py` → `tests/unit/test_graph.py`.

## Adding a dependency

1. Check stdlib first.
2. Add to the correct requirements file:
   - `requirements.txt` — runtime deps (both API + training).
   - `requirements-serve.txt` — serving-only (if the training pipeline doesn't need it).
   - `requirements-test.txt` — test-only.
3. Pin a minimum version (`pkg>=X.Y`).
4. Update `pyproject.toml` `[project.dependencies]` if the dep is a runtime one.

## Type checking

`mypy --strict` is wired as **warning-only** for now (Makefile target: `make typecheck`).
When you touch a module, aim to leave it strict-clean; we will flip the CI gate
on once the repo is entirely green.

## Quick reference — what Make can do

```bash
make help                     # lists every target with a one-liner
```
