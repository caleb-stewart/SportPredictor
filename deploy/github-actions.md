# GitHub Actions deployment

The manual workflow is [`.github/workflows/deploy-homelab.yml`](../.github/workflows/deploy-homelab.yml).
It runs on a GitHub-hosted runner, temporarily joins the Tailnet, syncs the
backend plus private Vue repository to Docker Ubuntu, and runs the existing
Compose deployment. It does **not** copy the host-side `.env`, database, model
artifacts, or reports, so those remain persistent on Docker Ubuntu.

This GitHub Actions workflow is separate from manual Codex devbox deployment.
Manual deployment uses the devbox's `~/.ssh/id_ed25519_devbox` key and the SSH
configuration documented in [`../docs/codex-devbox-runbook.md`](../docs/codex-devbox-runbook.md).

## One-time setup

Before the first GitHub-run deployment, commit and push the deployment files in
both repositories. The frontend repository must include `Dockerfile`,
`.dockerignore`, and `nginx/default.conf`.

Create these repository secrets in **SportPredictor → Settings → Secrets and
variables → Actions**:

| Secret | Value |
| --- | --- |
| `DEPLOY_SSH_PRIVATE_KEY` | A deployment private key stored only as this GitHub Actions secret. It is not required for manual Codex devbox deployment and is not stored in this repository. |
| `FRONTEND_REPO_TOKEN` | A fine-grained GitHub personal access token with **Contents: Read** access limited to `caleb-stewart/sport-predictor-vue`. |
| `TS_OAUTH_CLIENT_ID` | Tailscale OAuth client ID for the CI runner. |
| `TS_OAUTH_SECRET` | Tailscale OAuth client secret for the CI runner. |

Create the Tailscale OAuth client in the Tailscale admin console's **Trust
credentials** page. Give it the writable **Devices: Core** and **Auth keys**
scopes and permit the `tag:ci` tag. Your Tailnet policy must allow `tag:ci` to
reach `docker-ubuntu` on TCP port 22. A minimal rule is conceptually:

```json
{
  "tagOwners": { "tag:ci": ["autogroup:admin"] },
  "acls": [
    { "action": "accept", "src": ["tag:ci"], "dst": ["docker-ubuntu:22"] }
  ]
}
```

Adapt this to the rest of your current policy rather than replacing it.

## Run it

On GitHub, open **Actions → Deploy to Docker Ubuntu → Run workflow**. The
optional `frontend_ref` input defaults to `main`; use a branch, tag, or commit
SHA when you want to deploy a particular frontend revision.

The workflow uses the currently selected backend revision and reports a failure
unless the deployed Tailnet endpoint answers `GET /api/health`.
