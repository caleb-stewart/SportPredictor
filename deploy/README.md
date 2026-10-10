# Home-lab deployment

`deploy/deploy.sh` deploys the current working copies of this backend and its
sibling `../sport-predictor-vue` repository to Docker Ubuntu. It creates one
isolated Compose project under `/home/caleb/sportpredictor` on that host.

See [Home-lab access runbook](homelab-access.md) for the Tailscale/SSH
connection path, host layout, and Proxmox access boundary.

For the complete Codex devbox migration, local evaluation workflow, key setup,
recovery snapshots, and safety boundaries, see
[`../docs/codex-devbox-runbook.md`](../docs/codex-devbox-runbook.md).

The public Tailnet entrypoint is:

```text
http://docker-ubuntu.tail0da2aa.ts.net:8181
```

The browser calls `/api` on that same origin. FastAPI and PostgreSQL are not
published as host ports. Persistent PostgreSQL data, model artifacts, and
reports stay under `/home/caleb/sportpredictor/data` on Docker Ubuntu.

## First deploy

Ensure the local `.env` contains the HockeyTech and other API credentials, then
run:

```bash
./deploy/deploy.sh --seed-database
```

The script makes a fresh local PostgreSQL dump, transfers it over SSH, and uses
it only if the remote PostgreSQL data directory is new. The remote `.env` is
created once with a random database password and is mode `0600`.

## Later deploys

```bash
./deploy/deploy.sh
```

This syncs both working trees, rebuilds the `api` and `web` containers, runs
Alembic migrations through the API entrypoint, and checks `/api/health`.

To intentionally copy changed API credentials/settings from local `.env`:

```bash
./deploy/deploy.sh --refresh-env
```

To update the bootstrap dump for a potential future fresh database without
touching the already-initialized remote database:

```bash
./deploy/deploy.sh --seed-database
```

For the guarded, one-way process that replaces the live home-lab database with
the current local development database, see [Database data and sync](database-data.md).

## Host operations

```bash
ssh docker-ubuntu.tail0da2aa.ts.net
cd /home/caleb/sportpredictor
docker compose ps
docker compose logs -f api
docker compose logs -f web
```

Back up the running database before any destructive maintenance:

```bash
cd /home/caleb/sportpredictor
mkdir -p backups
docker compose exec -T db pg_dump -U sportpredictor sportpredictor > backups/sportpredictor_$(date +%F).sql
```
