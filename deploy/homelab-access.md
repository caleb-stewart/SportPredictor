# Home-lab access runbook

This project runs on the **Docker Ubuntu** guest inside the Proxmox home lab.
It is a Docker host, not the Proxmox hypervisor itself.

## Connection path

1. Connect the development machine to the Tailnet:

   ```bash
   tailscale status
   ```

   If it is logged out, run `tailscale login` and complete the browser sign-in.

2. Connect to Docker Ubuntu over Tailscale MagicDNS:

   ```bash
   ssh docker-ubuntu.tail0da2aa.ts.net
   ```

   On the Codex devbox, `~/.ssh/config` maps this host to
   `~/.ssh/id_ed25519_devbox`, and that public key is authorized for the
   `caleb` account. Use batch mode for automation:

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=15 \
     docker-ubuntu.tail0da2aa.ts.net 'hostname && docker compose version'
   ```

3. Manual deployment does not require a GitHub Actions private key. GitHub
   Actions uses its own repository secret, if that workflow is enabled:

   ```text
   DEPLOY_SSH_PRIVATE_KEY
   ```

   The value must remain only in GitHub Actions secrets; never commit or paste
   it into chat. The Codex devbox uses `id_ed25519_devbox` for manual deploys.

## Docker Ubuntu layout

| Purpose | Location / address |
| --- | --- |
| SportPredictor Compose project | `/home/caleb/sportpredictor` |
| Website on the Tailnet | `http://docker-ubuntu.tail0da2aa.ts.net:8181` |
| Persistent PostgreSQL data | `/home/caleb/sportpredictor/data/postgres` |
| Model artifacts | `/home/caleb/sportpredictor/data/model_store` |
| SportPredictor host secrets | `/home/caleb/sportpredictor/.env` (mode `0600`) |
| Existing media stack | `/docker/media/compose.yaml` |
| Retired Terraria world/config data | `/docker/terraria/data` |

Useful checks:

```bash
ssh docker-ubuntu.tail0da2aa.ts.net
cd /home/caleb/sportpredictor
docker compose ps
docker compose logs -f api
curl --fail http://localhost:8181/api/health
```

## Deployment

From a Tailnet-connected development machine, deploy the current local working
trees with:

```bash
./deploy/deploy.sh
```

Use `./deploy/deploy.sh --seed-database` only to refresh the bootstrap dump for
a future, brand-new PostgreSQL initialization. It does not overwrite an
already-initialized host database.

The manual GitHub Actions pipeline is documented in
[`github-actions.md`](github-actions.md). It reaches this same host through a
temporary Tailscale `tag:ci` node.

For database source-of-truth guidance, regular production ingestion, backups,
and the explicit local-to-home-lab restore procedure, see
[Database data and sync](database-data.md).

## Caleb Linux boundary

Caleb Linux is retired and is not part of the deployment path. Do not use its
SSH keys, working tree, database, or credentials for normal development or
deployment. The current code, local database snapshot, application `.env`
files, and Docker Ubuntu recovery snapshot are on the Codex devbox.

## Proxmox boundary

The Proxmox node's Tailnet name is `proxmox.tail0da2aa.ts.net`. From the Codex
devbox, `ssh proxmox` uses the separate `id_ed25519_devbox` key and is reserved
for VM administration. Do not assume that Docker credentials work on Proxmox
or store a Proxmox password in this repository.

To resize the Docker Ubuntu VM or change its CPU/RAM allocation, obtain a
separate Proxmox user/key with permission to manage that guest, identify its
VMID, inspect its current configuration, and then make the change through the
Proxmox UI or `qm`/`pct` on the hypervisor.
