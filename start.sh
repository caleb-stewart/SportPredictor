#!/usr/bin/env bash
set -euo pipefail

API_PORT="${API_PORT:-3141}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
FRONTEND_DIR="${FRONTEND_DIR:-../sport-predictor-vue}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ALEMBIC_BIN="${ALEMBIC_BIN:-.venv/bin/alembic}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}/"

API_PID=""
UI_PID=""

cleanup() {
  echo "[start.sh] Stopping running services..."
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
  if [[ -n "${UI_PID}" ]] && kill -0 "${UI_PID}" 2>/dev/null; then
    kill "${UI_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[start.sh] Missing Python runtime at ${PYTHON_BIN}. Create it with: python3 -m venv .venv"
  exit 1
fi

if [[ ! -x "${ALEMBIC_BIN}" ]]; then
  echo "[start.sh] Missing Alembic CLI at ${ALEMBIC_BIN}. Run: .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "[start.sh] Frontend directory not found: ${FRONTEND_DIR}"
  exit 1
fi

if ! command -v yarn >/dev/null 2>&1; then
  echo "[start.sh] yarn is required but not installed."
  exit 1
fi

echo "[start.sh] Running Alembic migrations..."
"${ALEMBIC_BIN}" upgrade head

echo "[start.sh] Starting FastAPI on port ${API_PORT}..."
"${PYTHON_BIN}" -m uvicorn main:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

echo "[start.sh] Starting Vue app on port ${FRONTEND_PORT}..."
(
  cd "${FRONTEND_DIR}"
  yarn dev --host --port "${FRONTEND_PORT}"
) &
UI_PID=$!

echo "[start.sh] API health: http://localhost:${API_PORT}/health"
echo "[start.sh] API docs:   http://localhost:${API_PORT}/docs"
echo "[start.sh] Frontend:   ${FRONTEND_URL}"

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${FRONTEND_URL}" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open "${FRONTEND_URL}" >/dev/null 2>&1 &
else
  echo "[start.sh] Cannot auto-open browser: neither 'xdg-open' nor 'open' is available."
  exit 1
fi

set +e
wait -n "${API_PID}" "${UI_PID}"
EXIT_CODE=$?
set -e

echo "[start.sh] A managed process exited unexpectedly (code=${EXIT_CODE})."
exit "${EXIT_CODE}"
