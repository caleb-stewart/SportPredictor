#!/usr/bin/env bash
# Replace the home-lab database with a verified dump of the local development
# database. A home-lab backup is made first and retained on Docker Ubuntu.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_HOST="${DEPLOY_HOST:-docker-ubuntu.tail0da2aa.ts.net}"
DEPLOY_USER="${DEPLOY_USER:-caleb}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/home/caleb/sportpredictor}"
SSH_TARGET="${DEPLOY_USER}@${DEPLOY_HOST}"
CONFIRMED=false

usage() {
  cat <<'EOF'
Usage: ./deploy/sync-local-database.sh --confirm

Creates a verified backup of the current home-lab database, then replaces it
with a verified custom-format dump of the local DATABASE_URL database.

This is intentionally one-way: local development database -> home-lab database.
It stops the home-lab API and web containers briefly while restoring data.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --confirm) CONFIRMED=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$CONFIRMED" != true ]]; then
  printf 'Refusing to replace the home-lab database without --confirm.\n' >&2
  usage >&2
  exit 2
fi

[[ -f "${BACKEND_DIR}/.env" ]] || { printf 'Missing %s/.env\n' "$BACKEND_DIR" >&2; exit 1; }
command -v pg_dump >/dev/null || { printf 'pg_dump is required locally.\n' >&2; exit 1; }
command -v pg_restore >/dev/null || { printf 'pg_restore is required locally.\n' >&2; exit 1; }

DATABASE_URL="$(sed -n 's/^DATABASE_URL=//p' "${BACKEND_DIR}/.env" | tail -n 1)"
[[ -n "$DATABASE_URL" ]] || { printf 'DATABASE_URL is not set in %s/.env\n' "$BACKEND_DIR" >&2; exit 1; }
LOCAL_DATABASE_URL="${DATABASE_URL/+psycopg2/}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_DUMP="$(mktemp --suffix=.dump sportpredictor-local-sync.XXXXXX)"
REMOTE_BACKUP="backups/home-lab-before-local-sync-${TIMESTAMP}.dump"
REMOTE_IMPORT="backups/local-import-${TIMESTAMP}.dump"

cleanup() {
  rm -f "$LOCAL_DUMP"
}
trap cleanup EXIT

printf 'Creating and verifying local database dump...\n'
pg_dump --dbname="$LOCAL_DATABASE_URL" --format=custom --no-owner --no-privileges --file="$LOCAL_DUMP"
pg_restore --list "$LOCAL_DUMP" >/dev/null

printf 'Backing up the current home-lab database...\n'
ssh -o BatchMode=yes "$SSH_TARGET" "
  set -eu
  cd '${DEPLOY_ROOT}'
  mkdir -p backups
  docker compose -f compose.yaml exec -T db sh -c 'pg_dump -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -Fc' > '${REMOTE_BACKUP}'
  test -s '${REMOTE_BACKUP}'
  docker compose -f compose.yaml exec -T db pg_restore --list < '${REMOTE_BACKUP}' >/dev/null
"

printf 'Uploading verified local dump...\n'
rsync -az "$LOCAL_DUMP" "${SSH_TARGET}:${DEPLOY_ROOT}/${REMOTE_IMPORT}"

printf 'Replacing the home-lab database and restarting the app...\n'
ssh -o BatchMode=yes "$SSH_TARGET" "
  set -eu
  cd '${DEPLOY_ROOT}'
  docker compose -f compose.yaml stop api web
  docker compose -f compose.yaml exec -T db sh -c 'dropdb -U \"\$POSTGRES_USER\" --maintenance-db=postgres --force \"\$POSTGRES_DB\"'
  docker compose -f compose.yaml exec -T db sh -c 'createdb -U \"\$POSTGRES_USER\" \"\$POSTGRES_DB\"'
  docker compose -f compose.yaml exec -T db sh -c 'pg_restore -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" --no-owner --no-privileges' < '${REMOTE_IMPORT}'
  docker compose -f compose.yaml up -d api web
  for attempt in \$(seq 1 24); do
    curl --fail --silent http://localhost:8181/api/health >/dev/null && exit 0
    sleep 5
  done
  docker compose -f compose.yaml ps
  docker compose -f compose.yaml logs --tail=100 api web
  exit 1
"

printf 'Database sync complete. Home-lab backup retained at %s/%s\n' "$DEPLOY_ROOT" "$REMOTE_BACKUP"
