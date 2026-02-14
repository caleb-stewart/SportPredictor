"""Export the canonical WHL Predictor V2 dataset to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from whl_v2_dataset import DbConfig, export_dataset_csv, load_canonical_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = str(Path(__file__).resolve().parent / "tmp" / "whl_v2_canonical_dataset.csv")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", default=None)
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db = DbConfig(
        host=args.db_host or DbConfig.host,
        port=args.db_port or DbConfig.port,
        dbname=args.db_name or DbConfig.dbname,
        user=args.db_user or DbConfig.user,
        password=args.db_password or DbConfig.password,
    )

    dataset = load_canonical_dataset(db)
    output_path = Path(args.output)
    export_dataset_csv(output_path, dataset)
    print(f"Exported {len(dataset)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
