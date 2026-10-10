#!/usr/bin/env bash
# Deploy the checked-out backend and sibling Vue repository to Docker Ubuntu.
# Secrets stay in the host-side .env file after the first deploy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${FRONTEND_DIR:-$(cd -- "${BACKEND_DIR}/../sport-predictor-vue" && pwd)}"
DEPLOY_HOST="${DEPLOY_HOST:-docker-ubuntu.tail0da2aa.ts.net}"
DEPLOY_USER="${DEPLOY_USER:-caleb}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/home/caleb/sportpredictor}"
SSH_TARGET="${DEPLOY_USER}@${DEPLOY_HOST}"
SEED_DATABASE=false
REFRESH_ENV=false

usage() {
  cat <<'EOF'
Usage: ./deploy/deploy.sh [--seed-database] [--refresh-env]

  --seed-database  Replace the host's bootstrap SQL with a fresh dump of the
                   local configured database. It affects PostgreSQL only if
                   the host database directory has not been initialized.
  --refresh-env     Replace the host .env with this checkout's .env. Use only
                   when API credentials or non-database settings have changed.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --seed-database) SEED_DATABASE=true ;;
    --refresh-env) REFRESH_ENV=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

require_file() {
  [[ -f "$1" ]] || { printf 'Missing required file: %s\n' "$1" >&2; exit 1; }
}

require_file "${BACKEND_DIR}/.env"
require_file "${BACKEND_DIR}/Dockerfile"
require_file "${FRONTEND_DIR}/Dockerfile"
require_file "${FRONTEND_DIR}/nginx/default.conf"

ssh -o BatchMode=yes "${SSH_TARGET}" "mkdir -p '${DEPLOY_ROOT}/backend' '${DEPLOY_ROOT}/frontend' '${DEPLOY_ROOT}/data/model_store' '${DEPLOY_ROOT}/data/reports' '${DEPLOY_ROOT}/data/postgres'"

if [[ "$REFRESH_ENV" == true ]] || ! ssh -o BatchMode=yes "${SSH_TARGET}" "test -f '${DEPLOY_ROOT}/.env'"; then
  rsync -az "${BACKEND_DIR}/.env" "${SSH_TARGET}:${DEPLOY_ROOT}/.env"
  ssh -o BatchMode=yes "${SSH_TARGET}" "
    set -eu
    cd '${DEPLOY_ROOT}'
    grep -q '^POSTGRES_USER=' .env || printf 'POSTGRES_USER=sportpredictor\\n' >> .env
    grep -q '^POSTGRES_PASSWORD=' .env || printf 'POSTGRES_PASSWORD=%s\\n' \"\$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)\" >> .env
    grep -q '^POSTGRES_DB=' .env || printf 'POSTGRES_DB=sportpredictor\\n' >> .env
    grep -q '^WEB_PORT=' .env || printf 'WEB_PORT=8181\\n' >> .env
    chmod 600 .env
  "
fi

if [[ "$SEED_DATABASE" == true ]] || ! ssh -o BatchMode=yes "${SSH_TARGET}" "test -s '${DEPLOY_ROOT}/data/bootstrap.sql'"; then
  dump_file="$(mktemp --suffix=.sql sportpredictor-bootstrap.XXXXXX)"
  trap 'rm -f "$dump_file"' EXIT
  DATABASE_URL="$(sed -n 's/^DATABASE_URL=//p' "${BACKEND_DIR}/.env" | tail -n 1)"
  [[ -n "$DATABASE_URL" ]] || { printf 'DATABASE_URL is not set in %s/.env\n' "$BACKEND_DIR" >&2; exit 1; }
  local_database_url="${DATABASE_URL/+psycopg2/}"
  pg_dump --dbname="$local_database_url" --format=plain --no-owner --no-privileges --file="$dump_file"
  rsync -az "$dump_file" "${SSH_TARGET}:${DEPLOY_ROOT}/data/bootstrap.sql"
  ssh -o BatchMode=yes "${SSH_TARGET}" "chmod 644 '${DEPLOY_ROOT}/data/bootstrap.sql'"
fi

rsync -az --delete \
  --exclude '.git/' --exclude '.env' --exclude '.venv/' --exclude '__pycache__/' \
  --exclude '.pytest_cache/' --exclude 'backups/' --exclude 'reports/' \
  --exclude 'model_store/' --exclude '*.sql' --exclude '*.dump' \
  "${BACKEND_DIR}/" "${SSH_TARGET}:${DEPLOY_ROOT}/backend/"

rsync -az --delete \
  --exclude '.git/' --exclude '.env' --exclude 'node_modules/' --exclude 'dist/' \
  "${FRONTEND_DIR}/" "${SSH_TARGET}:${DEPLOY_ROOT}/frontend/"

rsync -az "${SCRIPT_DIR}/compose.yaml" "${SSH_TARGET}:${DEPLOY_ROOT}/compose.yaml"
rsync -az "${BACKEND_DIR}/model_store/" "${SSH_TARGET}:${DEPLOY_ROOT}/data/model_store/"

ssh -o BatchMode=yes "${SSH_TARGET}" "
  set -eu
  cd '${DEPLOY_ROOT}'
  docker compose -f compose.yaml up -d --build --remove-orphans
  for attempt in \$(seq 1 24); do
    curl --fail --silent http://localhost:8181/api/health >/dev/null && exit 0
    sleep 5
  done
  docker compose -f compose.yaml ps
  docker compose -f compose.yaml logs --tail=100 api web
  exit 1
"

printf 'SportPredictor deployed: http://%s:8181\n' "$DEPLOY_HOST"
