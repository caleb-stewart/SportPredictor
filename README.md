# SportPredictor (Python-Only FastAPI Backend)

Hockey predictor backend (`whl`, `ohl`, `lhjmq`, `nhl`) using FastAPI + SQLAlchemy + APScheduler.

## Pre-Cutover Safety

Create a portable, checksummed backup before schema/runtime changes (this always uses the active `DATABASE_URL`, including the Compose port `5433`):

```bash
./.venv/bin/python -m scripts.backup_database --label pre_change
```

The script verifies every dump with `pg_restore --list`, writes a SHA-256 manifest next to it, and keeps the 14 newest dumps by default. Copy completed `backups/*.dump` and `backups/*.json` to storage outside this machine before relying on it for disaster recovery.

Restore the local Compose database without putting a password in shell history:

```bash
docker compose exec -T postgres sh -lc \
  'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-privileges' \
  < backups/sportpredictor_<timestamp>_<label>.dump
```

Capture baseline CHL table counts:

```bash
docker compose exec -T postgres psql \
  -U postgres -d sportpredictor_development -At -c "
SELECT 'chl_games|'||count(*) FROM chl_games UNION ALL
SELECT 'chl_teams|'||count(*) FROM chl_teams UNION ALL
SELECT 'chl_rolling_averages|'||count(*) FROM chl_rolling_averages UNION ALL
SELECT 'chl_prediction_records|'||count(*) FROM chl_prediction_records;"
```

## Setup

For the Codex devbox paths, SSH keys, local database, evaluation workflow,
recovery snapshots, and deployment procedure, read
[`docs/codex-devbox-runbook.md`](docs/codex-devbox-runbook.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The training pipeline uses the `psql` CLI to export canonical datasets, so PostgreSQL client tools must be installed.

## Local PostgreSQL with Docker Compose

The repository includes a PostgreSQL-only Compose service with a persistent named volume:

```bash
docker compose up -d postgres
docker compose ps
docker compose logs -f postgres
```

The host-run FastAPI app and Alembic commands use the active `DATABASE_URL` in
`.env`. Keep that value out of documentation and shell history. The Compose
service publishes on port `5433` by default so it can coexist with an existing
PostgreSQL server on `5432`.

On a fresh Docker volume, `3-28-2026-backup.sql` is imported automatically from the host-mounted file. Existing volumes are not re-imported.

Run migrations after the database becomes healthy:

```bash
./.venv/bin/alembic upgrade head
```

The database files remain in the `sportpredictor_postgres_data` Docker volume. Stop the container with `docker compose stop postgres`; normal container restarts, `docker compose down`, and host reboots retain this volume. `docker compose down -v`, Docker volume pruning, or Docker reset delete it, so use the portable backup command above before any such operation and copy the resulting backup off this machine.

## Environment

Required for CHL ingestion:
- `DATABASE_URL`
- `HOCKEYTECH_API_KEY_WHL|HOCKEYTECH_API_KEY_OHL|HOCKEYTECH_API_KEY_LHJMQ`

NHL ingestion uses the official public GameCenter API and official GS/FS reports; it does not require an NHL API key. The endpoints can be overridden with `NHL_API_BASE_URL` and `NHL_REPORT_BASE_URL`.

Recommended defaults are in `.env.example`.

Optional sportsbook data collection:

- `THE_ODDS_API_KEY` enables the database-backed Odds API collector.
- `THE_ODDS_API_CURRENT_SPORTS=icehockey_nhl`
- `THE_ODDS_API_REGIONS=us`
- `THE_ODDS_API_MARKETS=h2h,spreads,totals`
- `THE_ODDS_API_CURRENT_SYNC_INTERVAL_MINUTES=360` keeps the default NHL collection within a typical free-tier budget. The collector stores all returned books and outcomes as quote history.

Runtime flags:
- `DATA_BACKEND=chl`
- `DEFAULT_LEAGUE_CODE=whl|ohl|lhjmq|nhl`
- `LOCAL_MAINTENANCE_REFRESH_ENABLED=true|false` (manual league refresh API guard)
- `LOCAL_MAINTENANCE_MAX_CATCHUP_DAYS=14`

## Database Migrations (Alembic)

Existing DB cutover:

```bash
alembic stamp 20260214_180000
alembic upgrade head
```

Fresh DB:

```bash
alembic upgrade head
```

## Run Locally

Start backend + frontend with migrations and browser open:

```bash
./start.sh
```

VS Code launch (no bash script needed):

1. Open `SportPredictor` in VS Code.
2. Go to Run and Debug.
3. Run `Full Stack: API + UI`.

This runs Alembic migrations first, then starts FastAPI on `3141` and Vite on `5173`.

Direct API run:

```bash
uvicorn main:app --reload --port 3141
```

## API Endpoints

- `GET /health`
- `GET /teams?league_code=whl`
- `GET /games/upcoming?league_code=whl&date=YYYY-MM-DD`
- `POST /predictions/upcoming/run?league_code=whl&date=YYYY-MM-DD`
- `POST /predictions/custom`
- `GET /predictions/history?league_code=whl&date_from=&date_to=&team_id=&k_value=`
- `POST /predictions/replay/run` (body supports `league_code`)
- `GET /predictions/replay/report/{run_id}?league_code=whl`
- `POST /models/train?league_code=whl`
- `GET /models/active?league_code=whl`
- `GET /leagues`
- `GET /leagues/{league_code}/teams`
- `GET /leagues/{league_code}/predicted-games`
- `GET /leagues/{league_code}/predictions/results/by-day`
- `GET /leagues/{league_code}/reports/daily`
- `GET /leagues/{league_code}/reports/daily/{prediction_date}`
- `GET /leagues/{league_code}/games/{game_id}`
- `GET /leagues/{league_code}/maintenance/status`
- `GET /market-data/status`
- `POST /market-data/the-odds-api/sync/current?sport_key=icehockey_nhl`
- `POST /leagues/{league_code}/maintenance/refresh`
- `POST /models/compare/run?league_code=whl`
- `GET /models/compare/report/{run_id}?league_code=whl`
- `POST /experiments/feature-proposal/run?league_code=whl`
- `GET /experiments/{experiment_id}?league_code=whl`

## Scheduler (APScheduler)

Timezone: `America/Los_Angeles`

- `09:00` fetch next-day schedule
- `09:10` fetch yesterday updates
- `09:20` recompute rolling averages
- `09:30` train model (stage candidate)
- `09:40` promote staged model (if gates pass)
- `09:45` predict next-day games
- `Sunday 10:00` weekly model compare (`candidate=active` vs configured baseline)

Manual daily run:

```bash
python -m scripts.run_daily_pipeline
```

Sync current NHL moneylines, puck lines, and totals into PostgreSQL:

```bash
python -m scripts.sync_market_odds current
```

Historical snapshots require an Odds API plan that includes historical data. The importer is deliberately bounded to 25 daily requests by default so a command cannot unexpectedly exhaust quota:

```bash
python -m scripts.sync_market_odds history \
  --sport icehockey_nhl \
  --from 2025-10-01T00:00:00Z \
  --to 2026-04-30T00:00:00Z
```

Manual local catch-up refresh (league-scoped, guarded by `LOCAL_MAINTENANCE_REFRESH_ENABLED=true`):

```bash
curl -X POST "http://localhost:3141/leagues/ohl/maintenance/refresh"
curl "http://localhost:3141/leagues/ohl/maintenance/status"
```

Historical CHL backfill (season-based, resumable):

```bash
python -m scripts.backfill_chl_history --leagues ohl,lhjmq --dry-run
python -m scripts.backfill_chl_history --leagues ohl,lhjmq --all-available-seasons true --include-regular true --include-playoffs true --include-preseason true
```

NHL historical backfill (date-based; start with a small range and verify counts before expanding):

```bash
python -m scripts.backfill_nhl_history --date-from 2023-10-01 --date-to 2024-04-30 --dry-run
python -m scripts.backfill_nhl_history --date-from 2023-10-01 --date-to 2024-04-30 --rps-limit 3
# An interrupted run prints its run_id; resume it to retry recorded failures.
python -m scripts.backfill_nhl_history --date-from 2023-10-01 --date-to 2024-04-30 --resume-run-id <run_id>
# Full historical range used by this local project. It is safe to rerun because game writes are upserts.
./scripts/complete_nhl_history_backfill.sh
```

After a completed NHL backfill, the daily pipeline will build the same rolling features and train the first league-specific model:

```bash
python -m scripts.run_daily_pipeline --league-code nhl
```

The NHL adapter writes provider ids into the existing `hockeytech_id` columns for schema compatibility. The feature builder only treats these as league-scoped provider ids, so the existing rolling-average and prediction algorithm can consume NHL rows without a model/schema fork. NHL requires completed games with scores, shots, power-play opportunities/goals, and faceoff totals for complete training rows; the backfill reports any incomplete-stat games.

Freeze baseline proof snapshot:

```bash
python -m scripts.freeze_proof_baseline \
  --candidate-model-version 20260217T025309Z \
  --baseline-model-version 20260214T185052Z \
  --date-from 2023-09-01 \
  --date-to 2025-09-05 \
  --output reports/baseline_proof_snapshot.json
```

## Model Artifacts

Stored under `model_store/whl_v2/` with league subdirectories:
- versioned bundles (`model_bundle.joblib`, `metrics.json`, `metadata.json`)
- `active_model.json`
- `pending_model.json`

## Tests

```bash
pytest -q
```
