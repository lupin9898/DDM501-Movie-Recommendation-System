#!/usr/bin/env bash
# Generate strong random secrets and write them to .env.
#
# Usage:
#   ./scripts/gen-secrets.sh              # write to ./.env (refuses to overwrite)
#   ./scripts/gen-secrets.sh --force      # overwrite existing ./.env
#   ./scripts/gen-secrets.sh --stdout     # print to stdout only
#
# Generates:
#   POSTGRES_PASSWORD, MINIO_PASSWORD, GRAFANA_PASSWORD,
#   AIRFLOW_ADMIN_PASSWORD, AIRFLOW_FERNET_KEY, AIRFLOW_SECRET_KEY
#
# All other variables are copied from .env.example with their default values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_EXAMPLE="$REPO_ROOT/.env.example"
ENV_OUT="$REPO_ROOT/.env"

FORCE=0
STDOUT=0
for arg in "$@"; do
  case "$arg" in
    --force)   FORCE=1 ;;
    --stdout)  STDOUT=1 ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$ENV_EXAMPLE" ]]; then
  echo "ERROR: $ENV_EXAMPLE not found" >&2
  exit 1
fi

if [[ "$STDOUT" -eq 0 && -f "$ENV_OUT" && "$FORCE" -eq 0 ]]; then
  echo "ERROR: $ENV_OUT already exists. Re-run with --force to overwrite." >&2
  exit 1
fi

# ── Locate a Python 3 that has the cryptography module ───────────────────────
find_python() {
  for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c "import cryptography" >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(find_python || true)"
if [[ -z "$PY" ]]; then
  echo "ERROR: Python 3 with 'cryptography' package is required." >&2
  echo "       Install it via:  pip install cryptography" >&2
  exit 1
fi

# ── Generators ───────────────────────────────────────────────────────────────
gen_hex() {  # gen_hex <byte-length>
  "$PY" -c "import secrets; print(secrets.token_hex($1))"
}

gen_fernet() {
  "$PY" -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
}

POSTGRES_PASSWORD="$(gen_hex 16)"
MINIO_PASSWORD="$(gen_hex 16)"
GRAFANA_PASSWORD="$(gen_hex 16)"
AIRFLOW_ADMIN_PASSWORD="$(gen_hex 16)"
AIRFLOW_FERNET_KEY="$(gen_fernet)"
AIRFLOW_SECRET_KEY="$(gen_hex 32)"

# ── Render .env from .env.example, replacing "changeme" placeholders ────────
render_env() {
  "$PY" - "$ENV_EXAMPLE" <<PYEOF
import os, re, sys
example = open(sys.argv[1]).read()
subs = {
    "POSTGRES_PASSWORD":      os.environ["POSTGRES_PASSWORD"],
    "MINIO_PASSWORD":         os.environ["MINIO_PASSWORD"],
    "GRAFANA_PASSWORD":       os.environ["GRAFANA_PASSWORD"],
    "AIRFLOW_ADMIN_PASSWORD": os.environ["AIRFLOW_ADMIN_PASSWORD"],
    "AIRFLOW_FERNET_KEY":     os.environ["AIRFLOW_FERNET_KEY"],
    "AIRFLOW_SECRET_KEY":     os.environ["AIRFLOW_SECRET_KEY"],
}
def replace(match):
    key = match.group(1)
    if key in subs:
        return f"{key}={subs[key]}"
    return match.group(0)
rendered = re.sub(r"^(\w+)=changeme.*\$", replace, example, flags=re.MULTILINE)
# Also replace known plaintext dev keys that are not literal "changeme"
for key, val in subs.items():
    rendered = re.sub(
        rf"^{key}=.*\$",
        f"{key}={val}",
        rendered,
        flags=re.MULTILINE,
    )
print(rendered, end="")
PYEOF
}

export POSTGRES_PASSWORD MINIO_PASSWORD GRAFANA_PASSWORD \
       AIRFLOW_ADMIN_PASSWORD AIRFLOW_FERNET_KEY AIRFLOW_SECRET_KEY

if [[ "$STDOUT" -eq 1 ]]; then
  render_env
  exit 0
fi

render_env > "$ENV_OUT"
chmod 600 "$ENV_OUT"

echo "Wrote $ENV_OUT with freshly generated secrets (mode 600)."
echo "Do NOT commit this file."
