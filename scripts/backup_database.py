"""Create a portable, checksummed PostgreSQL backup of the active database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.engine import make_url

from scripts._cli_utils import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from core.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and verify a portable PostgreSQL backup.")
    parser.add_argument("--output-dir", type=Path, default=Path("backups"))
    parser.add_argument("--label", default="manual", help="Safe descriptive suffix for the output filename.")
    parser.add_argument("--keep", type=int, default=14, help="Keep this many newest custom dumps (0 keeps all).")
    return parser.parse_args()


def _safe_label(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value).strip("-_") or "backup"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prune(output_dir: Path, keep: int) -> list[str]:
    if keep <= 0:
        return []
    dumps = sorted(output_dir.glob("sportpredictor_*.dump"), key=lambda item: item.stat().st_mtime, reverse=True)
    removed: list[str] = []
    for dump in dumps[keep:]:
        manifest = dump.with_suffix(".json")
        dump.unlink()
        if manifest.exists():
            manifest.unlink()
        removed.append(dump.name)
    return removed


def main() -> None:
    args = parse_args()
    if args.keep < 0:
        raise SystemExit("--keep must be zero or greater")
    pg_dump = shutil.which("pg_dump")
    pg_restore = shutil.which("pg_restore")
    if not pg_dump or not pg_restore:
        raise SystemExit("PostgreSQL client tools pg_dump and pg_restore must be installed")

    url = make_url(settings.database_url)
    if not url.database:
        raise SystemExit("DATABASE_URL must name a database")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"sportpredictor_{timestamp}_{_safe_label(args.label)}.dump"
    environment = dict(os.environ)
    if url.password:
        environment["PGPASSWORD"] = str(url.password)

    command = [
        pg_dump,
        "--format=custom",
        "--no-owner",
        "--no-privileges",
        "--file",
        str(destination),
        "--host",
        url.host or "localhost",
        "--username",
        url.username or "postgres",
        "--dbname",
        url.database,
    ]
    if url.port:
        command.extend(["--port", str(url.port)])
    subprocess.run(command, check=True, env=environment)
    subprocess.run([pg_restore, "--list", str(destination)], check=True, stdout=subprocess.DEVNULL, env=environment)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "database": url.database,
        "host": url.host or "localhost",
        "port": url.port,
        "format": "postgresql-custom",
        "file": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "verified_with": "pg_restore --list",
    }
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    removed = _prune(args.output_dir, args.keep)
    print(json.dumps({**manifest, "pruned": removed}, indent=2))


if __name__ == "__main__":
    main()
