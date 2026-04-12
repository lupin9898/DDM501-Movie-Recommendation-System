# Multi-stage build for Movie Recommendation System

# Stage 1: Base with shared dependencies
FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Serving API
# Artifacts (model.pkl, mappings) are NOT baked in — they are mounted at
# runtime via the Docker volume defined in docker-compose.prod.yml.
# Training is handled separately by the train.yml GitHub Actions workflow.
FROM base AS server
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
