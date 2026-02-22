from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from sqlalchemy.orm import Session

from db.session import SessionLocal
from services.chl_history_ingest import _clock_has_core_fields, _merge_clock_with_gamesummary
from services.data_backend import hockeytech_api_key_for_league, hockeytech_client_code_for_league
from services.hockeytech_client import HockeyTechClient
from services.prediction_pipeline import build_clock_game_upsert_values

LOGGER = logging.getLogger("chl_games_repair")
FINAL_STATUS_TOKENS = {"4", "final", "completed", "game over", "end"}


@dataclass(frozen=True)
class _CandidateRow:
    game_id: int
    season_id: str | None
    season_name: str | None
    venue: str | None
    status: str | None
    home_goal_count: int | None
    away_goal_count: int | None
    scoring_breakdown: dict[str, Any] | None
    shots_on_goal: dict[str, Any] | None
    power_play: dict[str, Any] | None
    fow: dict[str, Any] | None
    home_power_play_percentage: Any
    away_power_play_percentage: Any
    home_faceoff_win_percentage: Any
    away_faceoff_win_percentage: Any
    home_shots_on_goal_total: int | None
    away_shots_on_goal_total: int | None
    scheduled_time_utc: Any
    scorebar_snapshot: dict[str, Any] | None


class _RateLimiter:
    def __init__(self, rps_limit: float) -> None:
        self._interval = (1.0 / rps_limit) if rps_limit > 0 else 0.0
        self._next_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._interval <= 0:
            return

        sleep_for = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                sleep_for = self._next_time - now
            self._next_time = max(now, self._next_time) + self._interval

        if sleep_for > 0:
            time.sleep(sleep_for)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _missing_text(value: Any) -> bool:
    return _normalized_text(value) == ""


def _is_final_status(value: Any) -> bool:
    token = _normalized_text(value).lower()
    if token in FINAL_STATUS_TOKENS:
        return True
    return "final" in token or "completed" in token


def _is_zero_like(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return True
        try:
            return float(stripped) == 0.0
        except ValueError:
            return False
    if isinstance(value, dict):
        if not value:
            return True
        return all(_is_zero_like(v) for v in value.values())
    if isinstance(value, list):
        if not value:
            return True
        return all(_is_zero_like(v) for v in value)
    return False


def _is_non_final_placeholder_row(row: _CandidateRow) -> bool:
    if _is_final_status(row.status):
        return False

    goals_zero_or_missing = (row.home_goal_count in (None, 0)) and (row.away_goal_count in (None, 0))
    if not goals_zero_or_missing:
        return False

    return all(
        [
            _is_zero_like(row.scoring_breakdown),
            _is_zero_like(row.shots_on_goal),
            _is_zero_like(row.power_play),
            _is_zero_like(row.fow),
            _is_zero_like(row.home_power_play_percentage),
            _is_zero_like(row.away_power_play_percentage),
            _is_zero_like(row.home_faceoff_win_percentage),
            _is_zero_like(row.away_faceoff_win_percentage),
            _is_zero_like(row.home_shots_on_goal_total),
            _is_zero_like(row.away_shots_on_goal_total),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and repair CHL games data quality issues.")
    parser.add_argument("--league-id", type=int, default=2)
    parser.add_argument("--mode", choices=["audit", "repair"], default="audit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--rps-limit", type=float, default=6.0)
    parser.add_argument("--infer-venue", type=_parse_bool, default=True)
    parser.add_argument("--reset-future-placeholders", type=_parse_bool, default=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _league_code_for_id(db: Session, league_id: int) -> str:
    code = db.execute(text("SELECT code FROM chl_leagues WHERE id=:league_id"), {"league_id": league_id}).scalar_one_or_none()
    if not code:
        raise RuntimeError(f"Unknown league_id={league_id}")
    return str(code)


def _audit_counts(db: Session, league_id: int) -> dict[str, Any]:
    final_sql = (
        "(LOWER(BTRIM(COALESCE(status,''))) IN ('final','completed','game over','end') "
        "OR BTRIM(COALESCE(status,''))='4')"
    )
    nonfinal_sql = f"(NOT {final_sql})"
    json_missing = lambda field: f"({field} IS NULL OR {field}::text='null')"  # noqa: E731

    q = text(
        f"""
        SELECT
          COUNT(*) AS total_rows,
          COUNT(*) FILTER (WHERE season_name IS NULL) AS season_name_sql_null,
          COUNT(*) FILTER (WHERE COALESCE(BTRIM(season_name),'')='') AS season_name_missing,
          COUNT(*) FILTER (WHERE venue IS NULL) AS venue_sql_null,
          COUNT(*) FILTER (WHERE COALESCE(BTRIM(venue),'')='') AS venue_missing,
          COUNT(*) FILTER (WHERE fow IS NULL) AS fow_sql_null,
          COUNT(*) FILTER (WHERE fow::text='null') AS fow_json_null,
          COUNT(*) FILTER (WHERE {json_missing('fow')}) AS fow_missing,
          COUNT(*) FILTER (WHERE home_goal_count IS NULL) AS home_goal_missing,
          COUNT(*) FILTER (WHERE away_goal_count IS NULL) AS away_goal_missing,
          COUNT(*) FILTER (WHERE {json_missing('scoring_breakdown')}) AS scoring_missing,
          COUNT(*) FILTER (WHERE {json_missing('shots_on_goal')}) AS shots_missing,
          COUNT(*) FILTER (WHERE {json_missing('power_play')}) AS power_play_missing,
          COUNT(*) FILTER (WHERE {final_sql}) AS final_rows,
          COUNT(*) FILTER (WHERE {nonfinal_sql}) AS nonfinal_rows,
          COUNT(*) FILTER (
            WHERE {final_sql}
              AND (
                home_goal_count IS NULL OR away_goal_count IS NULL
                OR {json_missing('scoring_breakdown')}
                OR {json_missing('shots_on_goal')}
                OR {json_missing('power_play')}
              )
          ) AS final_core_missing_rows,
          COUNT(*) FILTER (
            WHERE {nonfinal_sql}
              AND home_goal_count=0
              AND away_goal_count=0
          ) AS nonfinal_zero_goal_rows
        FROM chl_games
        WHERE league_id=:league_id
        """
    )
    row = db.execute(q, {"league_id": league_id}).mappings().one()
    return dict(row)


def _apply_updates(db: Session, league_id: int, game_id: int, updates: dict[str, Any], *, dry_run: bool) -> int:
    if not updates:
        return 0
    if dry_run:
        return 1

    allowed_fields = {
        "season_name",
        "venue",
        "status",
        "home_goal_count",
        "away_goal_count",
        "game_number",
        "period",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "scoring_breakdown",
        "shots_on_goal",
        "power_play",
        "fow",
        "home_power_play_percentage",
        "away_power_play_percentage",
        "home_faceoff_win_percentage",
        "away_faceoff_win_percentage",
        "home_shots_on_goal_total",
        "away_shots_on_goal_total",
        "scheduled_time_utc",
    }
    keys = [k for k in updates if k in allowed_fields]
    if not keys:
        return 0

    set_clause = ", ".join([f"{k}=:{k}" for k in keys] + ["updated_at=NOW()"])
    sql = text(f"UPDATE chl_games SET {set_clause} WHERE league_id=:league_id AND game_id=:game_id")
    params = {k: updates[k] for k in keys}
    params["league_id"] = league_id
    params["game_id"] = game_id
    db.execute(sql, params)
    return 1


def _repair_from_snapshot(db: Session, league_id: int, *, dry_run: bool) -> dict[str, Any]:
    sql = text(
        """
        UPDATE chl_games
        SET
          status = CASE
            WHEN (status IS NULL OR BTRIM(status)='' OR BTRIM(status) ~ '^[0-9]+$')
                 AND COALESCE(NULLIF(scorebar_snapshot->>'game_status',''), '') <> ''
              THEN scorebar_snapshot->>'game_status'
            ELSE status
          END,
          home_goal_count = COALESCE(
            home_goal_count,
            CASE
              WHEN COALESCE(scorebar_snapshot->>'home_goal_count', '') ~ '^-?[0-9]+$'
                THEN (scorebar_snapshot->>'home_goal_count')::int
              ELSE NULL
            END
          ),
          away_goal_count = COALESCE(
            away_goal_count,
            CASE
              WHEN COALESCE(scorebar_snapshot->>'visiting_goal_count', '') ~ '^-?[0-9]+$'
                THEN (scorebar_snapshot->>'visiting_goal_count')::int
              ELSE NULL
            END
          ),
          game_number = COALESCE(
            game_number,
            CASE
              WHEN COALESCE(scorebar_snapshot->>'game_number', '') ~ '^-?[0-9]+$'
                THEN (scorebar_snapshot->>'game_number')::int
              ELSE NULL
            END
          ),
          period = COALESCE(period, NULLIF(scorebar_snapshot->>'period', '')),
          home_team = COALESCE(home_team, NULLIF(scorebar_snapshot->>'home_team_name', '')),
          away_team = COALESCE(away_team, NULLIF(scorebar_snapshot->>'visiting_team_name', '')),
          home_team_id = COALESCE(
            home_team_id,
            CASE
              WHEN COALESCE(scorebar_snapshot->>'home_team', '') ~ '^-?[0-9]+$'
                THEN (scorebar_snapshot->>'home_team')::int
              ELSE NULL
            END
          ),
          away_team_id = COALESCE(
            away_team_id,
            CASE
              WHEN COALESCE(scorebar_snapshot->>'visiting_team', '') ~ '^-?[0-9]+$'
                THEN (scorebar_snapshot->>'visiting_team')::int
              ELSE NULL
            END
          ),
          venue = CASE
            WHEN (venue IS NULL OR BTRIM(venue)='') AND COALESCE(NULLIF(scorebar_snapshot->>'venue_name',''), '') <> ''
              THEN scorebar_snapshot->>'venue_name'
            ELSE venue
          END,
          scheduled_time_utc = COALESCE(
            scheduled_time_utc,
            CASE
              WHEN COALESCE(scorebar_snapshot->>'GameDateISO8601', '') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T'
                THEN (scorebar_snapshot->>'GameDateISO8601')::timestamptz
              ELSE NULL
            END
          ),
          updated_at = NOW()
        WHERE league_id=:league_id
          AND scorebar_snapshot IS NOT NULL
          AND (
            status IS NULL OR BTRIM(status)='' OR BTRIM(status) ~ '^[0-9]+$'
            OR home_goal_count IS NULL
            OR away_goal_count IS NULL
            OR game_number IS NULL
            OR period IS NULL
            OR home_team IS NULL
            OR away_team IS NULL
            OR home_team_id IS NULL
            OR away_team_id IS NULL
            OR venue IS NULL OR BTRIM(venue)=''
            OR scheduled_time_utc IS NULL
          )
        """
    )
    if dry_run:
        estimate = db.execute(
            text(
                """
                SELECT COUNT(*) FROM chl_games
                WHERE league_id=:league_id
                  AND scorebar_snapshot IS NOT NULL
                  AND (
                    status IS NULL OR BTRIM(status)='' OR BTRIM(status) ~ '^[0-9]+$'
                    OR home_goal_count IS NULL
                    OR away_goal_count IS NULL
                    OR game_number IS NULL
                    OR period IS NULL
                    OR home_team IS NULL
                    OR away_team IS NULL
                    OR home_team_id IS NULL
                    OR away_team_id IS NULL
                    OR venue IS NULL OR BTRIM(venue)=''
                    OR scheduled_time_utc IS NULL
                  )
                """
            ),
            {"league_id": league_id},
        ).scalar_one()
        return {"rows_touched": int(estimate), "dry_run": True}

    result = db.execute(sql, {"league_id": league_id})
    db.commit()
    return {"rows_touched": int(result.rowcount or 0), "dry_run": False}


def _season_name_map(client: HockeyTechClient) -> dict[str, str]:
    seasons = client.get_seasons()
    return {
        str(row.get("season_id")): str(row.get("season_name"))
        for row in seasons
        if row.get("season_id") not in (None, "") and row.get("season_name") not in (None, "")
    }


def _repair_season_names(db: Session, league_id: int, season_name_by_id: dict[str, str], *, dry_run: bool) -> dict[str, Any]:
    total = 0
    for season_id, season_name in season_name_by_id.items():
        if dry_run:
            touched = db.execute(
                text(
                    """
                    SELECT COUNT(*) FROM chl_games
                    WHERE league_id=:league_id
                      AND season_id=:season_id
                      AND COALESCE(BTRIM(season_name),'')=''
                    """
                ),
                {"league_id": league_id, "season_id": season_id},
            ).scalar_one()
            total += int(touched)
            continue

        result = db.execute(
            text(
                """
                UPDATE chl_games
                SET season_name=:season_name, updated_at=NOW()
                WHERE league_id=:league_id
                  AND season_id=:season_id
                  AND COALESCE(BTRIM(season_name),'')=''
                """
            ),
            {"league_id": league_id, "season_id": season_id, "season_name": season_name},
        )
        total += int(result.rowcount or 0)

    if not dry_run:
        db.commit()
    return {"rows_touched": total, "season_ids_seen": len(season_name_by_id), "dry_run": dry_run}


def _build_refetch_candidates(db: Session, league_id: int) -> list[_CandidateRow]:
    final_sql = (
        "(LOWER(BTRIM(COALESCE(status,''))) IN ('final','completed','game over','end') "
        "OR BTRIM(COALESCE(status,''))='4')"
    )
    q = text(
        f"""
        SELECT
          game_id,
          season_id,
          season_name,
          venue,
          status,
          home_goal_count,
          away_goal_count,
          scoring_breakdown,
          shots_on_goal,
          power_play,
          fow,
          home_power_play_percentage,
          away_power_play_percentage,
          home_faceoff_win_percentage,
          away_faceoff_win_percentage,
          home_shots_on_goal_total,
          away_shots_on_goal_total,
          scheduled_time_utc,
          scorebar_snapshot
        FROM chl_games
        WHERE league_id=:league_id
          AND (
            COALESCE(BTRIM(season_name),'')=''
            OR COALESCE(BTRIM(venue),'')=''
            OR (
              {final_sql}
              AND (
                home_goal_count IS NULL
                OR away_goal_count IS NULL
                OR scoring_breakdown IS NULL OR scoring_breakdown::text='null'
                OR shots_on_goal IS NULL OR shots_on_goal::text='null'
                OR power_play IS NULL OR power_play::text='null'
              )
            )
          )
        ORDER BY game_date DESC, game_id DESC
        """
    )
    rows = db.execute(q, {"league_id": league_id}).mappings().all()
    out: list[_CandidateRow] = []
    for row in rows:
        out.append(
            _CandidateRow(
                game_id=int(row["game_id"]),
                season_id=str(row["season_id"]) if row["season_id"] is not None else None,
                season_name=row["season_name"],
                venue=row["venue"],
                status=row["status"],
                home_goal_count=row["home_goal_count"],
                away_goal_count=row["away_goal_count"],
                scoring_breakdown=row["scoring_breakdown"],
                shots_on_goal=row["shots_on_goal"],
                power_play=row["power_play"],
                fow=row["fow"],
                home_power_play_percentage=row["home_power_play_percentage"],
                away_power_play_percentage=row["away_power_play_percentage"],
                home_faceoff_win_percentage=row["home_faceoff_win_percentage"],
                away_faceoff_win_percentage=row["away_faceoff_win_percentage"],
                home_shots_on_goal_total=row["home_shots_on_goal_total"],
                away_shots_on_goal_total=row["away_shots_on_goal_total"],
                scheduled_time_utc=row["scheduled_time_utc"],
                scorebar_snapshot=row["scorebar_snapshot"],
            )
        )
    return out


def _fillable_text(value: Any) -> str | None:
    token = _normalized_text(value)
    return token if token else None


def _should_fetch_gamesummary(row: _CandidateRow, clock_payload: dict[str, Any]) -> bool:
    if not _clock_has_core_fields(clock_payload):
        return True
    if _missing_text(row.season_name) and _missing_text(clock_payload.get("season_name")):
        return True
    if _missing_text(row.venue) and _missing_text(clock_payload.get("venue")):
        return True
    return False


def _repair_by_refetch(
    db: Session,
    *,
    league_id: int,
    league_code: str,
    season_name_by_id: dict[str, str],
    max_workers: int,
    rps_limit: float,
    dry_run: bool,
) -> dict[str, Any]:
    candidates = _build_refetch_candidates(db, league_id=league_id)
    if not candidates:
        return {"candidates": 0, "rows_updated": 0, "errors": []}

    limiter = _RateLimiter(rps_limit=rps_limit)
    thread_local = threading.local()
    errors: list[dict[str, Any]] = []
    rows_updated = 0

    def _get_client() -> HockeyTechClient:
        cached = getattr(thread_local, "client", None)
        if cached is not None:
            return cached
        client = HockeyTechClient(
            client_code=hockeytech_client_code_for_league(league_code),
            api_key=hockeytech_api_key_for_league(league_code),
        )
        thread_local.client = client
        return client

    def _worker(row: _CandidateRow) -> tuple[_CandidateRow, dict[str, Any] | None, str | None]:
        try:
            client = _get_client()
            limiter.wait()
            clock_payload = client.get_clock(row.game_id)
            if not clock_payload:
                return row, None, "Empty clock payload"

            if _should_fetch_gamesummary(row, clock_payload):
                limiter.wait()
                gamesummary = client.get_gamesummary(row.game_id)
                clock_payload = _merge_clock_with_gamesummary(clock_payload, gamesummary)

            return row, clock_payload, None
        except Exception as exc:  # noqa: BLE001
            return row, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_worker, row) for row in candidates]

        for idx, future in enumerate(as_completed(futures), start=1):
            row, clock_payload, err = future.result()
            if err or clock_payload is None:
                errors.append({"game_id": row.game_id, "error": err or "unknown"})
                continue

            game_data = row.scorebar_snapshot or {
                "id": str(row.game_id),
                "game_id": str(row.game_id),
                "season_id": row.season_id,
            }
            values, _ = build_clock_game_upsert_values(game_data=game_data, clock=clock_payload, now=time_now_utc())

            # Season name fallback map is deterministic and cheaper than another refetch.
            if _missing_text(values.get("season_name")) and row.season_id in season_name_by_id:
                values["season_name"] = season_name_by_id[row.season_id]

            updates: dict[str, Any] = {}
            if _missing_text(row.season_name):
                season_name_value = _fillable_text(values.get("season_name"))
                if season_name_value:
                    updates["season_name"] = season_name_value
            if _missing_text(row.venue):
                venue_value = _fillable_text(values.get("venue"))
                if venue_value:
                    updates["venue"] = venue_value

            if row.home_goal_count is None and values.get("home_goal_count") is not None:
                updates["home_goal_count"] = values["home_goal_count"]
            if row.away_goal_count is None and values.get("away_goal_count") is not None:
                updates["away_goal_count"] = values["away_goal_count"]

            if row.scoring_breakdown is None and values.get("scoring_breakdown") is not None:
                updates["scoring_breakdown"] = values["scoring_breakdown"]
            if row.shots_on_goal is None and values.get("shots_on_goal") is not None:
                updates["shots_on_goal"] = values["shots_on_goal"]
            if row.power_play is None and values.get("power_play") is not None:
                updates["power_play"] = values["power_play"]
            if row.fow is None and values.get("fow") is not None:
                updates["fow"] = values["fow"]

            if row.home_shots_on_goal_total is None and values.get("home_shots_on_goal_total") is not None:
                updates["home_shots_on_goal_total"] = values["home_shots_on_goal_total"]
            if row.away_shots_on_goal_total is None and values.get("away_shots_on_goal_total") is not None:
                updates["away_shots_on_goal_total"] = values["away_shots_on_goal_total"]

            if row.home_power_play_percentage is None and values.get("home_power_play_percentage") is not None:
                updates["home_power_play_percentage"] = values["home_power_play_percentage"]
            if row.away_power_play_percentage is None and values.get("away_power_play_percentage") is not None:
                updates["away_power_play_percentage"] = values["away_power_play_percentage"]
            if row.home_faceoff_win_percentage is None and values.get("home_faceoff_win_percentage") is not None:
                updates["home_faceoff_win_percentage"] = values["home_faceoff_win_percentage"]
            if row.away_faceoff_win_percentage is None and values.get("away_faceoff_win_percentage") is not None:
                updates["away_faceoff_win_percentage"] = values["away_faceoff_win_percentage"]

            if row.scheduled_time_utc is None and values.get("scheduled_time_utc") is not None:
                updates["scheduled_time_utc"] = values["scheduled_time_utc"]

            rows_updated += _apply_updates(
                db,
                league_id=league_id,
                game_id=row.game_id,
                updates=updates,
                dry_run=dry_run,
            )

            if not dry_run and idx % 100 == 0:
                db.commit()
                LOGGER.info("Refetch progress %s/%s rows_updated=%s errors=%s", idx, len(candidates), rows_updated, len(errors))

    if not dry_run:
        db.commit()
    return {"candidates": len(candidates), "rows_updated": rows_updated, "errors": errors}


def _venue_maps(db: Session, league_id: int) -> tuple[dict[tuple[int, str], str], dict[int, str]]:
    by_team_season: dict[tuple[int, str], str] = {}
    rows = db.execute(
        text(
            """
            SELECT home_team_id, season_id, venue
            FROM (
              SELECT home_team_id, season_id, venue, COUNT(*) AS c,
                     ROW_NUMBER() OVER (
                       PARTITION BY home_team_id, season_id
                       ORDER BY COUNT(*) DESC, venue ASC
                     ) AS rn
              FROM chl_games
              WHERE league_id=:league_id
                AND home_team_id IS NOT NULL
                AND season_id IS NOT NULL
                AND COALESCE(BTRIM(venue),'') <> ''
              GROUP BY home_team_id, season_id, venue
            ) ranked
            WHERE rn=1
            """
        ),
        {"league_id": league_id},
    ).fetchall()
    for home_team_id, season_id, venue in rows:
        by_team_season[(int(home_team_id), str(season_id))] = str(venue)

    by_team: dict[int, str] = {}
    rows2 = db.execute(
        text(
            """
            SELECT home_team_id, venue
            FROM (
              SELECT home_team_id, venue, COUNT(*) AS c,
                     ROW_NUMBER() OVER (
                       PARTITION BY home_team_id
                       ORDER BY COUNT(*) DESC, venue ASC
                     ) AS rn
              FROM chl_games
              WHERE league_id=:league_id
                AND home_team_id IS NOT NULL
                AND COALESCE(BTRIM(venue),'') <> ''
              GROUP BY home_team_id, venue
            ) ranked
            WHERE rn=1
            """
        ),
        {"league_id": league_id},
    ).fetchall()
    for home_team_id, venue in rows2:
        by_team[int(home_team_id)] = str(venue)
    return by_team_season, by_team


def _infer_missing_venues(db: Session, league_id: int, *, dry_run: bool) -> dict[str, Any]:
    by_team_season, by_team = _venue_maps(db, league_id=league_id)
    rows = db.execute(
        text(
            """
            SELECT game_id, home_team_id, season_id
            FROM chl_games
            WHERE league_id=:league_id
              AND COALESCE(BTRIM(venue),'')=''
            """
        ),
        {"league_id": league_id},
    ).fetchall()

    updated = 0
    unresolved = 0
    for game_id, home_team_id, season_id in rows:
        if home_team_id is None:
            unresolved += 1
            continue

        venue = None
        if season_id is not None:
            venue = by_team_season.get((int(home_team_id), str(season_id)))
        if not venue:
            venue = by_team.get(int(home_team_id))
        if not venue:
            unresolved += 1
            continue

        updated += _apply_updates(
            db,
            league_id=league_id,
            game_id=int(game_id),
            updates={"venue": venue},
            dry_run=dry_run,
        )

    if not dry_run:
        db.commit()
    return {"rows_updated": updated, "rows_unresolved": unresolved, "dry_run": dry_run}


def _placeholder_candidate_rows(db: Session, league_id: int) -> list[_CandidateRow]:
    final_sql = (
        "(LOWER(BTRIM(COALESCE(status,''))) IN ('final','completed','game over','end') "
        "OR BTRIM(COALESCE(status,''))='4')"
    )
    q = text(
        f"""
        SELECT
          game_id,
          season_id,
          season_name,
          venue,
          status,
          home_goal_count,
          away_goal_count,
          scoring_breakdown,
          shots_on_goal,
          power_play,
          fow,
          home_power_play_percentage,
          away_power_play_percentage,
          home_faceoff_win_percentage,
          away_faceoff_win_percentage,
          home_shots_on_goal_total,
          away_shots_on_goal_total,
          scheduled_time_utc,
          scorebar_snapshot
        FROM chl_games
        WHERE league_id=:league_id
          AND NOT {final_sql}
          AND (
            home_goal_count IS NOT NULL
            OR away_goal_count IS NOT NULL
            OR scoring_breakdown IS NOT NULL OR scoring_breakdown::text='null'
            OR shots_on_goal IS NOT NULL OR shots_on_goal::text='null'
            OR power_play IS NOT NULL OR power_play::text='null'
            OR fow IS NOT NULL OR fow::text='null'
            OR home_power_play_percentage IS NOT NULL
            OR away_power_play_percentage IS NOT NULL
            OR home_faceoff_win_percentage IS NOT NULL
            OR away_faceoff_win_percentage IS NOT NULL
            OR home_shots_on_goal_total IS NOT NULL
            OR away_shots_on_goal_total IS NOT NULL
          )
        """
    )
    out: list[_CandidateRow] = []
    for row in db.execute(q, {"league_id": league_id}).mappings().all():
        out.append(
            _CandidateRow(
                game_id=int(row["game_id"]),
                season_id=str(row["season_id"]) if row["season_id"] is not None else None,
                season_name=row["season_name"],
                venue=row["venue"],
                status=row["status"],
                home_goal_count=row["home_goal_count"],
                away_goal_count=row["away_goal_count"],
                scoring_breakdown=row["scoring_breakdown"],
                shots_on_goal=row["shots_on_goal"],
                power_play=row["power_play"],
                fow=row["fow"],
                home_power_play_percentage=row["home_power_play_percentage"],
                away_power_play_percentage=row["away_power_play_percentage"],
                home_faceoff_win_percentage=row["home_faceoff_win_percentage"],
                away_faceoff_win_percentage=row["away_faceoff_win_percentage"],
                home_shots_on_goal_total=row["home_shots_on_goal_total"],
                away_shots_on_goal_total=row["away_shots_on_goal_total"],
                scheduled_time_utc=row["scheduled_time_utc"],
                scorebar_snapshot=row["scorebar_snapshot"],
            )
        )
    return out


def _reset_nonfinal_placeholders(db: Session, league_id: int, *, dry_run: bool) -> dict[str, Any]:
    rows = _placeholder_candidate_rows(db, league_id=league_id)
    updated = 0
    for row in rows:
        if not _is_non_final_placeholder_row(row):
            continue
        updates = {
            "home_goal_count": None,
            "away_goal_count": None,
            "scoring_breakdown": None,
            "shots_on_goal": None,
            "power_play": None,
            "fow": None,
            "home_power_play_percentage": None,
            "away_power_play_percentage": None,
            "home_faceoff_win_percentage": None,
            "away_faceoff_win_percentage": None,
            "home_shots_on_goal_total": None,
            "away_shots_on_goal_total": None,
        }
        updated += _apply_updates(
            db,
            league_id=league_id,
            game_id=row.game_id,
            updates=updates,
            dry_run=dry_run,
        )

    if not dry_run:
        db.commit()
    return {"candidate_rows": len(rows), "rows_updated": updated, "dry_run": dry_run}


def time_now_utc():
    import datetime as dt

    return dt.datetime.now(dt.UTC)


def _run_repair(
    db: Session,
    *,
    league_id: int,
    max_workers: int,
    rps_limit: float,
    infer_venue: bool,
    reset_future_placeholders: bool,
    dry_run: bool,
) -> dict[str, Any]:
    league_code = _league_code_for_id(db, league_id)
    client = HockeyTechClient(
        client_code=hockeytech_client_code_for_league(league_code),
        api_key=hockeytech_api_key_for_league(league_code),
    )
    season_name_by_id = _season_name_map(client)

    before = _audit_counts(db, league_id)
    LOGGER.info("Audit before repair: %s", before)

    pass_a = _repair_from_snapshot(db, league_id=league_id, dry_run=dry_run)
    LOGGER.info("Pass A (snapshot fill): %s", pass_a)

    pass_b = _repair_season_names(db, league_id=league_id, season_name_by_id=season_name_by_id, dry_run=dry_run)
    LOGGER.info("Pass B (season name map): %s", pass_b)

    pass_c = _repair_by_refetch(
        db,
        league_id=league_id,
        league_code=league_code,
        season_name_by_id=season_name_by_id,
        max_workers=max_workers,
        rps_limit=rps_limit,
        dry_run=dry_run,
    )
    LOGGER.info(
        "Pass C (refetch) candidates=%s rows_updated=%s errors=%s",
        pass_c.get("candidates"),
        pass_c.get("rows_updated"),
        len(pass_c.get("errors", [])),
    )

    pass_d = {"skipped": True}
    if infer_venue:
        pass_d = _infer_missing_venues(db, league_id=league_id, dry_run=dry_run)
        LOGGER.info("Pass D (venue inference): %s", pass_d)

    pass_e = {"skipped": True}
    if reset_future_placeholders:
        pass_e = _reset_nonfinal_placeholders(db, league_id=league_id, dry_run=dry_run)
        LOGGER.info("Pass E (placeholder reset): %s", pass_e)

    after = _audit_counts(db, league_id)
    LOGGER.info("Audit after repair: %s", after)

    return {
        "league_id": league_id,
        "league_code": league_code,
        "dry_run": dry_run,
        "before": before,
        "pass_a_snapshot_fill": pass_a,
        "pass_b_season_name_fill": pass_b,
        "pass_c_refetch": {
            "candidates": pass_c.get("candidates", 0),
            "rows_updated": pass_c.get("rows_updated", 0),
            "error_count": len(pass_c.get("errors", [])),
            "errors_sample": pass_c.get("errors", [])[:25],
        },
        "pass_d_venue_inference": pass_d,
        "pass_e_reset_placeholders": pass_e,
        "after": after,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if args.log_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    db = SessionLocal()
    try:
        if args.mode == "audit":
            result = {
                "league_id": args.league_id,
                "league_code": _league_code_for_id(db, args.league_id),
                "audit": _audit_counts(db, args.league_id),
            }
            print(json.dumps(result, indent=2, default=str))
            return

        result = _run_repair(
            db,
            league_id=args.league_id,
            max_workers=args.max_workers,
            rps_limit=args.rps_limit,
            infer_venue=bool(args.infer_venue),
            reset_future_placeholders=bool(args.reset_future_placeholders),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(result, indent=2, default=str))
    finally:
        db.close()


if __name__ == "__main__":
    main()
