"""Train the V4 dynamic/hierarchical prototype without promoting it."""

from _cli_utils import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from services.v4_training import main


if __name__ == "__main__":
    raise SystemExit(main())
