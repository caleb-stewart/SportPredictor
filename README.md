# SportPredictor (Python-Only FastAPI Backend)

Single-backend WHL predictor using FastAPI + SQLAlchemy + APScheduler on the existing Postgres schema.

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

Capture baseline table counts:

```bash
PGPASSWORD=qqqq psql -h localhost -U postgres -d sportpredictor_development -At -c "\
SELECT 'whl_games|'||count(*) FROM whl_games UNION ALL \
SELECT 'whl_teams|'||count(*) FROM whl_teams UNION ALL \
SELECT 'whl_rolling_averages|'||count(*) FROM whl_rolling_averages UNION ALL \
SELECT 'whl_prediction_records|'||count(*) FROM whl_prediction_records;"
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
- `HOCKEYTECH_API`

Recommended defaults are in `.env.example`.

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
- `GET /teams`
- `GET /games/upcoming?date=YYYY-MM-DD`
- `POST /predictions/upcoming/run?date=YYYY-MM-DD`
- `POST /predictions/custom`
- `GET /predictions/history?date_from=&date_to=&team_id=&k_value=`
- `POST /models/train`
- `GET /models/active`

## Scheduler (APScheduler)

Timezone: `America/Los_Angeles`

- `09:00` fetch next-day schedule
- `09:10` fetch yesterday updates
- `09:20` recompute rolling averages
- `09:30` train model (stage candidate)
- `09:40` promote staged model (if gates pass)
- `09:45` predict next-day games

Manual daily run:

```bash
python scripts/run_daily_pipeline.py
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
