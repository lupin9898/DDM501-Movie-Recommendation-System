#!/bin/bash
# Create mlflow and airflow databases/users.
# Idempotent: creates users if missing, always syncs password to $POSTGRES_PASSWORD.
set -e

create_user_and_db() {
    local user=$1
    local db=$2

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${user}') THEN
                CREATE USER ${user} WITH PASSWORD '${POSTGRES_PASSWORD}';
            ELSE
                ALTER USER ${user} WITH PASSWORD '${POSTGRES_PASSWORD}';
            END IF;
        END
        \$\$;
        SELECT 'CREATE DATABASE ${db} OWNER ${user}'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${db}')\gexec
        GRANT ALL PRIVILEGES ON DATABASE ${db} TO ${user};
EOSQL
}

create_user_and_db mlflow mlflow
create_user_and_db airflow airflow

echo "Databases and users ready: mlflow, airflow"
