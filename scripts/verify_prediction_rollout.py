from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text, select, and_

from db.models import ChlGame, ChlPredictionRecord
from db.session import SessionLocal
from services.feature_builder import build_features_by_k
from services.leagues import get_game_detail, list_predicted_games
from services.predictor import get_active_model_pointer_metadata, get_model_status, predict_from_payload


def _check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def verify_rollout(
    *,
    league_code: str,
    expected_model_version: str | None,
    previous_model_version: str | None,
    previous_rows: int | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    status = get_model_status(league_code)
    pointer = get_active_model_pointer_metadata(league_code)
    active_version = status.get("model_version")

    _check(checks, "active_pointer_is_healthy", bool(status.get("active")), status)
    _check(
        checks,
        "active_version_matches_expected",
        expected_model_version is None or active_version == expected_model_version,
        {"expected": expected_model_version, "actual": active_version},
    )
    _check(
        checks,
        "active_pointer_bundle_exists",
        Path(str(pointer["bundle_path"])).exists(),
        pointer,
    )

    with SessionLocal() as db:
        migration = db.execute(text("select version_num from alembic_version")).scalars().all()
        _check(
            checks,
            "append_only_migration_is_applied",
            "20261010_030000" in migration,
            migration,
        )

        constraint_names = db.execute(
            text(
                """
                select conname
                from pg_constraint
                where conrelid = 'chl_prediction_records'::regclass
                  and contype = 'u'
                """
            )
        ).scalars().all()
        _check(
            checks,
            "prediction_uniqueness_includes_model_version",
            "uq_chl_prediction_records_league_game_k_model" in constraint_names,
            constraint_names,
        )

        version_rows = db.execute(
            text(
                """
                select model_version, model_family, count(*) as rows, count(distinct game_id) as games
                from chl_prediction_records
                where league_id = 1
                group by model_version, model_family
                order by model_version
                """
            )
        ).mappings().all()
        version_rows = [dict(row) for row in version_rows]
        active_row = next((row for row in version_rows if row["model_version"] == active_version), None)
        _check(checks, "active_rows_exist", active_row is not None and int(active_row["rows"]) > 0, active_row)

        incomplete_games = db.execute(
            text(
                """
                select count(*)
                from (
                    select game_id
                    from chl_prediction_records
                    where league_id = 1 and model_version = :model_version
                    group by game_id
                    having count(distinct k_value) <> 3
                ) incomplete
                """
            ),
            {"model_version": active_version},
        ).scalar_one()
        _check(
            checks,
            "every_active_game_has_all_rolling_rows",
            int(incomplete_games) == 0,
            {"incomplete_games": int(incomplete_games)},
        )

        previous_count = None
        if previous_model_version:
            previous_count = db.execute(
                text(
                    "select count(*) from chl_prediction_records where league_id=1 and model_version=:model_version"
                ),
                {"model_version": previous_model_version},
            ).scalar_one()
            _check(
                checks,
                "previous_rows_preserved",
                int(previous_count) > 0 and (previous_rows is None or int(previous_count) == previous_rows),
                {"model_version": previous_model_version, "rows": int(previous_count), "expected_rows": previous_rows},
            )

        scoring = db.execute(
            text(
                """
                select
                    count(*) filter (where ensemble_correct is not null) as resolved_rows,
                    count(*) filter (where ensemble_correct is true) as correct_rows,
                    avg(case when ensemble_correct is not null then (ensemble_correct::int)::numeric end) as accuracy
                from chl_prediction_records
                where league_id=1 and model_version=:model_version
                """
            ),
            {"model_version": active_version},
        ).mappings().one()
        _check(
            checks,
            "active_rows_have_canonical_ensemble_scoring",
            int(scoring["resolved_rows"] or 0) > 0,
            {key: float(value) if key == "accuracy" and value is not None else int(value or 0) for key, value in scoring.items()},
        )

        overlap = None
        if previous_model_version:
            overlap = db.execute(
                text(
                    """
                    with old_rows as (
                        select game_id, ensemble_correct
                        from chl_prediction_records
                        where league_id=1 and model_version=:old_version and k_value=15
                    ), new_rows as (
                        select game_id, ensemble_correct
                        from chl_prediction_records
                        where league_id=1 and model_version=:new_version and k_value=15
                    )
                    select
                        count(*) as overlapping_games,
                        avg(case when old_rows.ensemble_correct is not null then old_rows.ensemble_correct::int end) as old_accuracy,
                        avg(case when new_rows.ensemble_correct is not null then new_rows.ensemble_correct::int end) as new_accuracy
                    from old_rows join new_rows using (game_id)
                    """
                ),
                {"old_version": previous_model_version, "new_version": active_version},
            ).mappings().one()
            overlap = {
                "overlapping_games": int(overlap["overlapping_games"] or 0),
                "old_accuracy": float(overlap["old_accuracy"]) if overlap["old_accuracy"] is not None else None,
                "new_accuracy": float(overlap["new_accuracy"]) if overlap["new_accuracy"] is not None else None,
            }

        sample_game = db.scalar(
            select(ChlGame)
            .join(
                ChlPredictionRecord,
                and_(
                    ChlPredictionRecord.league_id == ChlGame.league_id,
                    ChlPredictionRecord.game_id == ChlGame.game_id,
                    ChlPredictionRecord.model_version == active_version,
                ),
            )
            .where(ChlGame.league_id == 1)
            .order_by(ChlGame.game_date.desc(), ChlGame.game_id.desc())
        )
        if sample_game is None:
            _check(checks, "website_sample_game_is_available", False, None)
        else:
            service_rows = list_predicted_games(
                db,
                league_code,
                prediction_date_from=None,
                prediction_date_to=None,
                game_date_from=sample_game.game_date,
                game_date_to=sample_game.game_date,
                team_provider_id=None,
                result="all",
                limit=5000,
                offset=0,
                sort="game_date_desc",
                include_k_breakdown=True,
            )
            service_game = next((row for row in service_rows if int(row["game_id"]) == int(sample_game.game_id)), None)
            detail = get_game_detail(db, league_code, int(sample_game.game_id))
            visible_versions = {
                prediction.get("model_version")
                for prediction in (service_game or {}).get("predictions_by_k", [])
            }
            detail_versions = {prediction.get("model_version") for prediction in detail.get("predictions_by_k", [])}
            _check(
                checks,
                "website_service_payload_is_active_only",
                service_game is not None and visible_versions == {active_version} and detail_versions == {active_version},
                {
                    "sample_game_id": int(sample_game.game_id),
                    "list_versions": sorted(visible_versions),
                    "detail_versions": sorted(detail_versions),
                },
            )

            k15 = next((row for row in (service_game or {}).get("predictions_by_k", []) if row["k_value"] == 15), None)
            _check(
                checks,
                "website_consensus_uses_stored_ensemble",
                service_game is not None
                and k15 is not None
                and abs(float(service_game["consensus_home_probability"]) - float(k15["ensemble_home_team_probability"])) < 0.0002,
                {
                    "consensus_home_probability": service_game.get("consensus_home_probability") if service_game else None,
                    "k15_ensemble_home_probability": k15.get("ensemble_home_team_probability") if k15 else None,
                },
            )

            built = build_features_by_k(
                db,
                home_team_hockeytech_id=int(sample_game.home_team_id),
                away_team_hockeytech_id=int(sample_game.away_team_id),
                game_date=sample_game.game_date,
                league_code=league_code,
            )
            live_payload = {
                "game_id": int(sample_game.game_id),
                "game_date": sample_game.game_date.isoformat(),
                "home_team_id": int(sample_game.home_team_id),
                "away_team_id": int(sample_game.away_team_id),
                "features_by_k": built["features_by_k"],
                "context_features": built.get("context_features") or {},
            }
            live_result = predict_from_payload(live_payload, league_code=league_code)
            _check(
                checks,
                "live_prediction_path_uses_active_model",
                live_result.get("model_version") == active_version and live_result.get("model_family") == status.get("model_family"),
                {"model_version": live_result.get("model_version"), "model_family": live_result.get("model_family")},
            )

    report = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "league_code": league_code,
        "active_model": status,
        "pointer": pointer,
        "version_rows": version_rows,
        "overlap_accuracy": overlap,
        "checks": checks,
        "overall_passed": all(check["passed"] for check in checks),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify an active prediction-model rollout end to end.")
    parser.add_argument("--league-code", default="whl")
    parser.add_argument("--expected-model-version")
    parser.add_argument("--previous-model-version", default="20260217T025309Z")
    parser.add_argument("--previous-rows", type=int, default=7242)
    parser.add_argument("--output", default="reports/verification/v3_rollout_verification.json")
    args = parser.parse_args()

    result = verify_rollout(
        league_code=args.league_code,
        expected_model_version=args.expected_model_version,
        previous_model_version=args.previous_model_version,
        previous_rows=args.previous_rows,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['name']}: {json.dumps(check['details'], default=str)}")
    print(json.dumps({"overall_passed": result["overall_passed"], "report": str(output_path)}, indent=2))
    raise SystemExit(0 if result["overall_passed"] else 1)
