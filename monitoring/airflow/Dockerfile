FROM apache/airflow:2.10.2
# lightfm needs C compilation (Cython) — install build tools as root
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Pre-create artifacts dir with correct ownership so training can write model.pkl
RUN mkdir -p /opt/airflow/artifacts && chown -R airflow:root /opt/airflow/artifacts
USER airflow
# Install project training dependencies (serving deps excluded)
COPY requirements.txt /tmp/training-requirements.txt
RUN pip install --no-cache-dir -r /tmp/training-requirements.txt
