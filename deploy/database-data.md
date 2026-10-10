# Database data and sync

## Normal production updates

The home-lab PostgreSQL database is independent from the local development
database. A normal code deploy does not copy local rows to Docker Ubuntu.

The production API starts with `SCHEDULER_ENABLED=true`, so it performs its own
daily provider ingestion, feature recomputation, training, and prediction jobs.
For ordinary day-to-day updates, treat the home-lab database as the production
source of truth and let that scheduler update it. Check its status with:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net
cd /home/caleb/sportpredictor
docker compose logs --since=24h api
curl --fail http://localhost:8181/api/health
```

The deployed scheduler's timing and pipeline behavior are described in the
project README and application logs. The host-side API credentials are stored
only in `/home/caleb/sportpredictor/.env`.

## When to copy local data to the home lab

Use a local-to-home-lab sync only when local development contains intentional
data changes that should become production data, such as a completed historical
backfill, a repair, or an imported data set that has been checked locally.

Because this replaces the whole target database, deploy the matching code first
and do not use it merely to obtain routine daily results.

```bash
./deploy/deploy.sh
./deploy/sync-local-database.sh --confirm
```

The sync script performs these safety steps:

1. Dumps the local `DATABASE_URL` database in PostgreSQL custom format.
2. Verifies that dump with `pg_restore --list`.
3. Creates and verifies a full custom-format backup of the current home-lab
   database at `/home/caleb/sportpredictor/backups/`.
4. Stops only the SportPredictor API and web containers.
5. Recreates the home-lab application database, restores the verified local
   dump, starts the API/web containers, and waits for `/api/health`.

The old home-lab database is retained as a timestamped backup. If a restore
fails, the API may remain stopped; do not run another destructive command.
Use the retained backup with `pg_restore`, or ask for help restoring it.

## Initial data versus ongoing sync

`./deploy/deploy.sh --seed-database` only changes the bootstrap SQL used for a
**future fresh** PostgreSQL initialization. It does not change the database in
the already-running home-lab stack.

`./deploy/sync-local-database.sh --confirm` is the explicit operation that
replaces the already-running home-lab database.

## Backups

Create a portable local backup before broad local changes:

```bash
./.venv/bin/python -m scripts.backup_database --label before-data-change
```

Create a remote backup without replacing anything:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net '
  cd /home/caleb/sportpredictor &&
  mkdir -p backups &&
  docker compose exec -T db sh -c '\''pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc'\'' \
    > backups/sportpredictor_$(date -u +%Y%m%dT%H%M%SZ).dump
'
```

Do not commit dumps, `.env` files, SSH keys, or backup manifests containing
private infrastructure details to GitHub.
