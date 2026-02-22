# SportPredictor (Python-Only FastAPI Backend)

CHL predictor backend (`whl`, `ohl`, `lhjmq`) using FastAPI + SQLAlchemy + APScheduler.

## Pre-Cutover Safety

Create a backup before schema/runtime changes:

```bash
mkdir -p backups
ts=$(date -u +%Y%m%dT%H%M%SZ)
PGPASSWORD=qqqq pg_dump -h localhost -U postgres -d sportpredictor_development -Fc -f backups/pre_cutover_${ts}.dump
```

Restore example:

```bash
PGPASSWORD=qqqq pg_restore -h localhost -U postgres -d sportpredictor_development --clean --if-exists backups/pre_cutover_<timestamp>.dump
```

Capture baseline CHL table counts:

```bash
PGPASSWORD=qqqq psql -h localhost -U postgres -d sportpredictor_development -At -c "\
SELECT 'chl_games|'||count(*) FROM chl_games UNION ALL \
SELECT 'chl_teams|'||count(*) FROM chl_teams UNION ALL \
SELECT 'chl_rolling_averages|'||count(*) FROM chl_rolling_averages UNION ALL \
SELECT 'chl_prediction_records|'||count(*) FROM chl_prediction_records;"
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The training pipeline uses the `psql` CLI to export canonical datasets, so PostgreSQL client tools must be installed.

## Environment

Required:
- `DATABASE_URL`
- `HOCKEYTECH_API_KEY_WHL|HOCKEYTECH_API_KEY_OHL|HOCKEYTECH_API_KEY_LHJMQ`

Recommended defaults are in `.env.example`.

Runtime flags:
- `DATA_BACKEND=chl`
- `DEFAULT_LEAGUE_CODE=whl|ohl|lhjmq`

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
- `GET /leagues/{league_code}/reports/daily`
- `GET /leagues/{league_code}/reports/daily/{prediction_date}`
- `GET /leagues/{league_code}/games/{game_id}`
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
python scripts/run_daily_pipeline.py
```

Historical CHL backfill (season-based, resumable):

```bash
python scripts/backfill_chl_history.py --leagues ohl,lhjmq --dry-run
python scripts/backfill_chl_history.py --leagues ohl,lhjmq --all-available-seasons true --include-regular true --include-playoffs true --include-preseason true
```

Freeze baseline proof snapshot:

```bash
python scripts/freeze_proof_baseline.py \
  --candidate-model-version 20260217T025309Z \
  --baseline-model-version 20260214T185052Z \
  --date-from 2023-09-01 \
  --date-to 2025-09-05 \
  --output reports/baseline_proof_snapshot.json
```

## Model Artifacts

Stored under `model_store/whl_v2/`:
- versioned bundles (`model_bundle.joblib`, `metrics.json`, `metadata.json`)
- `active_model.json`
- `pending_model.json`

## Tests

```bash
pytest -q
```
