#!/bin/bash
# Create multiple databases (mlflow and airflow) with the same password as postgres superuser
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD '$POSTGRES_PASSWORD';
    CREATE DATABASE mlflow OWNER mlflow;

    CREATE USER airflow WITH PASSWORD '$POSTGRES_PASSWORD';
    CREATE DATABASE airflow OWNER airflow;
EOSQL
