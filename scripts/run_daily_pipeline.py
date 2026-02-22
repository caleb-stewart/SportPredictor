from __future__ import annotations

import argparse
import json

from db.session import SessionLocal
from services.prediction_pipeline import run_daily_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily prediction pipeline.")
    parser.add_argument("--league-code", default="whl", help="League code (whl|ohl|lhjmq).")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = run_daily_pipeline(db, league_code=args.league_code)
        print(json.dumps(result, indent=2, default=str))
    finally:
        db.close()
