# Codex Devbox Runbook

This is the machine-specific operating guide for SportPredictor development,
evaluation, backup, and deployment. It is deliberately free of passwords,
API-key values, private keys, and database connection secrets.

## 1. Environment map

The Codex devbox is the primary development machine. Caleb Linux is retired;
future work must not require it to be online.

| Purpose | Location / command |
| --- | --- |
| Backend | `/home/caleb/Projects/SportPredictor` |
| Frontend | `/home/caleb/Projects/sport-predictor-vue` |
| Backend Python environment | `/home/caleb/Projects/SportPredictor/.venv` |
| Local PostgreSQL | Docker Compose service `postgres` |
| Local PostgreSQL host port | `5433` |
| Local database | `sportpredictor_development` |
| Local database volume | `sportpredictor_sportpredictor_postgres_data` |
| Deployment target | Docker Ubuntu guest at `docker-ubuntu.tail0da2aa.ts.net` |
| Deployment user | `caleb` |
| Deployment project | `/home/caleb/sportpredictor` |
| Production website | `http://docker-ubuntu.tail0da2aa.ts.net:8181` |
| Proxmox hypervisor | `proxmox` / VM 100 (`Docker-Ubuntu-Server`) |

The devbox currently has Docker 29.x, Compose, Python 3.12, Node 22, `uv`,
PostgreSQL 16 client tools, Git, `rsync`, Tailscale, and authenticated GitHub
CLI access.

## 2. SSH and deployment keys

The devbox key is:

```text
/home/caleb/.ssh/id_ed25519_devbox
```

`/home/caleb/.ssh/config` already maps this key to both the Docker Ubuntu host
and the Proxmox host. The Docker Ubuntu ED25519 host key was verified from both
the devbox and Caleb Linux before it was added to `known_hosts`.

Verify the two independent connections:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net 'hostname; id -un; docker compose version'
ssh proxmox 'hostname; id -un'
```

The first command is the normal app deployment connection. The second is for
Proxmox VM administration only. Do not assume that Docker Ubuntu credentials
are valid on Proxmox, or that Proxmox access is required for an ordinary app
deployment.

The devbox public key was added to Docker Ubuntu's `caleb` account. The private
key remains only on the devbox; Caleb Linux is not needed for deployment.

Git operations use the devbox's existing authenticated GitHub CLI session over
HTTPS. The repository remotes are HTTPS, so no GitHub private SSH key is
needed for normal source control.

Never put passwords, private keys, API tokens, or `.env` contents in this
document, Git, shell history, or chat.

## 3. Migrated files and recovery snapshots

The backend and frontend working trees were copied from Caleb Linux with their
uncommitted and untracked project work preserved. Disposable `.venv`,
`node_modules`, caches, and bytecode were rebuilt locally.

Application configuration is present at:

```text
/home/caleb/Projects/SportPredictor/.env
/home/caleb/Projects/sport-predictor-vue/.env
```

Both files are mode `0600`. Do not replace them with values pasted into chat.

The local development database snapshot from Caleb Linux is:

```text
/home/caleb/Projects/SportPredictor/backups/sportpredictor_20260924T022508Z_from_caleb-linux.dump
```

It was verified with `pg_restore --list`. It represents the database state at
the time of migration; it is not a live sync.

A separate recovery snapshot of the Docker Ubuntu deployment is stored outside
the Git repositories:

```text
/home/caleb/.local/share/sportpredictor/docker-ubuntu-snapshot-20261003/
```

It contains:

- `docker-ubuntu.env` — production host environment, mode `0600`
- `docker-ubuntu-sportpredictor.dump` — verified PostgreSQL custom-format dump, mode `0600`
- `compose.yaml` — deployed Compose definition
- `model_store/` — deployed model artifacts
- `reports/` — deployed reports

The production dump checksum at creation time was:

```text
c0521489e676ac8dc9e29a645cc422fc6ed064ba40407187f51417970731c375
```

These snapshots are point-in-time recovery copies. Refresh them deliberately
after a meaningful production change; do not overwrite them blindly.

## 4. Start local development

From the backend:

```bash
cd /home/caleb/Projects/SportPredictor
docker compose up -d postgres
docker compose ps
./.venv/bin/alembic current
```

Run the backend:

```bash
cd /home/caleb/Projects/SportPredictor
./.venv/bin/uvicorn main:app --reload --host 127.0.0.1 --port 3141
```

Run the frontend in a second terminal:

```bash
cd /home/caleb/Projects/sport-predictor-vue
npm run dev -- --host 127.0.0.1
```

The repository's `start.sh` and VS Code launch configuration can also start
the full local workflow. The local API uses PostgreSQL on `localhost:5433`.

## 5. Evaluate changes

Basic local health check:

```bash
curl --fail http://127.0.0.1:3141/health
```

Useful read-only API checks:

```bash
curl --fail 'http://127.0.0.1:3141/leagues'
curl --fail 'http://127.0.0.1:3141/leagues/whl/teams'
curl --fail 'http://127.0.0.1:3141/leagues/whl/predicted-games'
curl --fail 'http://127.0.0.1:3141/market-data/status'
```

Run the backend tests and frontend production build before deploying:

```bash
cd /home/caleb/Projects/SportPredictor
./.venv/bin/pytest -q

cd /home/caleb/Projects/sport-predictor-vue
npm run build
```

The migrated baseline passed 91 backend tests and a Vite production build.

For a production health check:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net \
  'cd /home/caleb/sportpredictor && docker compose ps && curl --fail --silent http://localhost:8181/api/health'
```

The production browser endpoint is:

```text
http://docker-ubuntu.tail0da2aa.ts.net:8181
```

## 6. Backups and database safety

Create a local portable backup before schema, backfill, repair, or broad data
changes:

```bash
cd /home/caleb/Projects/SportPredictor
./.venv/bin/python -m scripts.backup_database --label before-change
```

The backup script verifies the archive with `pg_restore --list` and writes a
checksum manifest. Keep important backups outside the Git repository and off
the devbox as well.

Back up the running Docker Ubuntu database without changing the application:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net '
  set -eu
  cd /home/caleb/sportpredictor
  mkdir -p backups
  docker compose exec -T db sh -lc '\''pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc'\'' \
    > backups/sportpredictor_$(date -u +%Y%m%dT%H%M%SZ).dump
'
```

Normal code deployment does not copy local rows to production. The production
database has its own scheduler and is normally the production source of truth.

`./deploy/deploy.sh --seed-database` only prepares bootstrap data for a future
fresh PostgreSQL initialization. It does not replace an already initialized
production database.

`./deploy/sync-local-database.sh --confirm` is destructive: it backs up and
then replaces the live Docker Ubuntu database with the local database. Use it
only for an intentional checked local data release, after deploying matching
code and confirming the backup exists.

Do not use `docker compose down -v` or Docker volume pruning without a verified
portable backup. The local database volume contains the migrated development
data.

## 7. Deploy the application

The deployment script syncs the current backend and sibling frontend working
trees to `/home/caleb/sportpredictor`, rebuilds the API and web containers,
runs the API entrypoint migrations, and checks the health endpoint.

Before deployment:

```bash
cd /home/caleb/Projects/SportPredictor
git status --short --branch
./.venv/bin/pytest -q
cd ../sport-predictor-vue
npm run build
cd ../SportPredictor
```

Normal code deployment:

```bash
./deploy/deploy.sh
```

The script uses the SSH configuration documented above. It excludes local
`.env`, backups, reports, model artifacts, virtualenvs, and Git metadata from
the code sync; those remain persistent on the Docker Ubuntu host.

Only use `--refresh-env` when intentionally changing production API credentials
or runtime settings:

```bash
./deploy/deploy.sh --refresh-env
```

After deployment:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net \
  'cd /home/caleb/sportpredictor && docker compose ps && docker compose logs --tail=100 api web'
```

Do not run a deployment merely to obtain current production data. Use the
backup procedure above instead.

## 8. Scheduler and data operations

The deployed API normally runs with its scheduler enabled. The scheduler owns
routine provider ingestion, feature recomputation, model training, and daily
predictions. Inspect recent production activity with:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net \
  'cd /home/caleb/sportpredictor && docker compose logs --since=24h api'
```

Manual and historical operations are documented in the backend README and
should be run with the appropriate league/date scope:

- `scripts.run_daily_pipeline`
- `scripts.sync_market_odds`
- `scripts.backfill_chl_history`
- `scripts.backfill_nhl_history`
- `scripts.complete_nhl_history_backfill.sh`

Start historical operations with a dry run or bounded date range, verify row
counts, and create a backup first.

## 9. Troubleshooting boundaries

- If local API health fails, check `docker compose ps`, `.env` mode `0600`, and
  that PostgreSQL is listening on `5433`.
- If production SSH fails, verify Tailscale status and the devbox SSH config;
  do not fall back to Caleb Linux.
- If production containers fail, inspect `docker compose logs api web` on Docker
  Ubuntu and preserve the database backup before changing data.
- Proxmox VM administration is separate from application deployment. The
  Docker Ubuntu guest is VM 100; it has no QEMU guest agent, so normal app
  operations use SSH to Docker Ubuntu.
- A plaintext credential was found in the Proxmox VM metadata during migration.
  Rotate it and remove it from the VM description; never copy it into this
  repository or this runbook.
