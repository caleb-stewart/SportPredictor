# Repository contribution workflow

## Branches and commits

- Never make application or infrastructure changes directly on `production` or `main`.
- Before editing, create a new focused branch from the latest intended base.
- Use a descriptive branch name such as `feature/nhl-support`, `fix/prediction-api`, or `chore/deployment-tooling`.
- Keep unrelated worktree changes untouched and out of the commit.
- Commit each cohesive change on its feature branch with a clear commit message.
- Run the relevant tests or checks before handing off the branch.
- Do not commit `.env` files, credentials, private keys, database dumps, or generated caches.

## Production changes

- Production deployments must use a clean checkout of the committed branch.
- Verify the branch, commit, migration state, service health, and relevant API behavior after deployment.
- Preserve append-only prediction data and create a verified database backup before destructive data operations.
