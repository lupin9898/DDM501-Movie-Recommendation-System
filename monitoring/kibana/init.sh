#!/bin/sh
# Wait for Kibana to be healthy, then import saved objects (data view + saved searches).
# Idempotent: re-imports overwrite existing objects with the same IDs.

set -eu

KIBANA_URL="${KIBANA_URL:-http://kibana:5601}"
SAVED_OBJECTS_FILE="${SAVED_OBJECTS_FILE:-/provisioning/saved-objects.ndjson}"

echo "[kibana-init] Waiting for Kibana at $KIBANA_URL ..."
until curl -fsS "$KIBANA_URL/api/status" > /dev/null 2>&1; do
  sleep 5
done

echo "[kibana-init] Kibana ready. Importing saved objects from $SAVED_OBJECTS_FILE"
curl -fsS -X POST "$KIBANA_URL/api/saved_objects/_import?overwrite=true" \
  -H "kbn-xsrf: true" \
  --form file=@"$SAVED_OBJECTS_FILE" \
  && echo "[kibana-init] Import complete." \
  || echo "[kibana-init] Import failed (non-fatal)."
