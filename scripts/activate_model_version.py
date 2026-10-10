from __future__ import annotations

import argparse
import json

from services.predictor import activate_model_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atomically activate a verified model version.")
    parser.add_argument("--league-code", default="whl")
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            activate_model_version(args.model_version, league_code=args.league_code),
            indent=2,
            default=str,
        )
    )
