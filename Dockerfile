# Multi-stage build for Movie Recommendation System

# Stage 1: Base with dependencies
FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Training (optional, for CI)
FROM base AS trainer
COPY src/ src/
COPY data/ data/
RUN python -m src.training.train --model als || echo "Training skipped (no data)"

# Stage 3: Serving
FROM base AS server
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
COPY --from=trainer /app/artifacts/ artifacts/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
