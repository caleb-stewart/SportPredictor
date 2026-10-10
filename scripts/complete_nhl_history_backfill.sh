#!/usr/bin/env bash
set -euo pipefail

# An optional PID lets this full backfill wait for an already-running legacy
# importer. New imports acquire a PostgreSQL advisory lock themselves.
wait_for_pid="${1:-}"
if [[ -n "${wait_for_pid}" ]]; then
  while kill -0 "${wait_for_pid}" 2>/dev/null; do
    sleep 60
  done
fi

python_bin="./.venv/bin/python"

# Keep the import resumable and bounded. The importer is idempotent, so a
# restarted block safely rehydrates existing rows and fills any missing stats.
run_range() {
  local date_from="$1"
  local date_to="$2"
  echo "Starting NHL backfill ${date_from} through ${date_to}"
  "${python_bin}" -m scripts.backfill_nhl_history \
    --date-from "${date_from}" --date-to "${date_to}" \
    --include-playoffs true --include-preseason false --rps-limit 3 --log-level INFO
}

# Official GameCenter history is available from the 2000-01 season onward.
run_range 2000-09-01 2017-08-31
run_range 2017-09-01 2023-09-30
run_range 2023-10-01 2024-09-30
run_range 2024-10-01 2025-09-30
run_range 2025-10-01 2026-09-20

"${python_bin}" -m scripts.backup_database --label after_nhl_history_backfill
