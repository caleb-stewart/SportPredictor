from __future__ import annotations

import json

from db.session import SessionLocal
from services.prediction_pipeline import run_daily_pipeline


if __name__ == "__main__":
    db = SessionLocal()
    try:
        result = run_daily_pipeline(db)
        print(json.dumps(result, indent=2, default=str))
    finally:
        db.close()
