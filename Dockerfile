# Multi-stage build for Movie Recommendation System

# Stage 1: Base with serving dependencies
# We install requirements-serve.txt (subset of requirements.txt) here so the
# production image stays lean — no mlflow/requests bloat. Training deps live
# in requirements.txt and are used by airflow/train workflow only.
FROM python:3.11-slim AS base
WORKDIR /app
# build-essential + python3-dev — needed to compile lightfm (Cython) on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Stage 2: Serving API
# Artifacts (model.pkl, mappings) are NOT baked in — they are mounted at
# runtime via the Docker volume defined in docker-compose.prod.yml.
# Training is handled separately by the train.yml GitHub Actions workflow.
FROM base AS server
COPY src/ src/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
