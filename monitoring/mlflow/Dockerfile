FROM ghcr.io/mlflow/mlflow:v2.12.1

# PostgreSQL driver + S3/MinIO support (not included in base image)
RUN pip install --no-cache-dir psycopg2-binary boto3

EXPOSE 5000
