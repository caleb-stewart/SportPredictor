"""Append-only historical prediction materialization.

This is the reusable task used when a model becomes active. It writes a
versioned set of prediction rows and never deletes or replaces another model's
history. Re-running the same version is idempotent.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import settings
from db.models import ChlGame, ChlPredictionRecord, ChlRollingAverage, ChlTeam
from services.context_features import build_context_feature_map
from services.data_backend import CHL_LEAGUES, require_supported_league_code
from services.feature_builder import InsufficientHistoryError, TeamNotFoundError, build_features_by_k
from services.prediction_pipeline import K_VALUES, _persist_prediction_rows
from services.predictor import ModelNotAvailableError, load_model_bundle_by_version, predict_from_payload
from services.training_features import build_pairwise_features


class MaterializationError(RuntimeError):
    pass


def _feature_map(row: ChlRollingAverage) -> dict[str, float]:
    return {
        "goals_for_avg": float(row.goals_for_avg or 0.0),
        "goals_against_avg": float(row.goals_against_avg or 0.0),
        "shots_for_avg": float(row.shots_for_avg or 0.0),
        "shots_against_avg": float(row.shots_against_avg or 0.0),
        "power_play_percentage_avg": float(row.power_play_percentage_avg or 0.0),
        "power_play_percentage_against_avg": float(row.power_play_percentage_against_avg or 0.0),
        "faceoff_win_percentage_avg": float(row.faceoff_win_percentage_avg or 0.0),
        "faceoff_win_percentage_against_avg": float(row.faceoff_win_percentage_against_avg or 0.0),
        "goals_diff": float(row.goals_diff or 0.0),
        "ppp_diff": float(row.ppp_diff or 0.0),
        "sog_diff": float(row.sog_diff or 0.0),
        "fowp_diff": float(row.fowp_diff or 0.0),
    }


def _load_context_map(db: Session, league_id: int) -> dict[int, dict[str, float]]:
    rows = db.execute(
        select(
            ChlGame.game_id,
            ChlGame.game_date,
            ChlGame.home_team_id,
            ChlGame.away_team_id,
            ChlGame.home_goal_count,
            ChlGame.away_goal_count,
        )
        .where(
            ChlGame.league_id == league_id,
            ChlGame.game_date.is_not(None),
            ChlGame.home_team_id.is_not(None),
            ChlGame.away_team_id.is_not(None),
            ChlGame.home_goal_count.is_not(None),
            ChlGame.away_goal_count.is_not(None),
        )
        .order_by(ChlGame.game_date.asc(), ChlGame.game_id.asc())
    ).mappings().all()
    if not rows:
        return {}
    return build_context_feature_map(pd.DataFrame([dict(row) for row in rows]))


def _load_rolling_snapshots(db: Session, league_id: int) -> dict[tuple[int, int, int], ChlRollingAverage]:
    rows = db.scalars(
        select(ChlRollingAverage).where(ChlRollingAverage.league_id == league_id)
    ).all()
    return {(int(row.game_id), int(row.k_value), int(row.home_away or 0)): row for row in rows}


def _fast_v3_payload(
    game: ChlGame,
    *,
    home_team_db_id: int,
    away_team_db_id: int,
    rolling: dict[tuple[int, int, int], ChlRollingAverage],
    context_by_game_id: dict[int, dict[str, float]],
) -> dict[str, Any]:
    features_by_k: dict[str, dict[str, dict[str, float]]] = {}
    for k in K_VALUES:
        home_row = rolling.get((int(game.game_id), k, 1))
        away_row = rolling.get((int(game.game_id), k, 0))
        if home_row is None or away_row is None:
            raise InsufficientHistoryError(f"Missing rolling snapshot for game={game.game_id}, k={k}.")
        features_by_k[str(k)] = {
            "home": _feature_map(home_row),
            "away": _feature_map(away_row),
        }

    context_features = dict(context_by_game_id.get(int(game.game_id)) or {})
    k15 = features_by_k["15"]
    engineered = build_pairwise_features(k15["home"], k15["away"])
    elo_diff = float(context_features.get("elo_diff_pre", 0.0))
    context_features["strength_adjusted_goals_diff"] = float(engineered["goals_diff_diff"]) - elo_diff / 400.0
    context_features["strength_adjusted_sog_diff"] = float(engineered["sog_diff_diff"]) - elo_diff / 400.0

    return {
        "game_id": int(game.game_id),
        "game_date": game.game_date.isoformat() if game.game_date else None,
        "home_team_id": int(game.home_team_id),
        "away_team_id": int(game.away_team_id),
        "features_by_k": features_by_k,
        "context_features": context_features,
    }


def _existing_game_ids(db: Session, league_id: int, model_version: str) -> set[int]:
    rows = db.execute(
        select(ChlPredictionRecord.game_id)
        .where(
            ChlPredictionRecord.league_id == league_id,
            ChlPredictionRecord.model_version == model_version,
            ChlPredictionRecord.k_value.in_(K_VALUES),
        )
        .group_by(ChlPredictionRecord.game_id)
        .having(func.count(func.distinct(ChlPredictionRecord.k_value)) == len(K_VALUES))
    ).scalars().all()
    return {int(game_id) for game_id in rows}


def materialize_model_version(
    db: Session,
    *,
    league_code: str,
    model_version: str,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
    batch_size: int = 250,
    skip_existing: bool = True,
    dry_run: bool = False,
    allow_scheduler: bool = False,
) -> dict[str, Any]:
    """Materialize all completed predictions for an explicit model version."""

    normalized = require_supported_league_code(league_code)
    if settings.scheduler_enabled and not allow_scheduler:
        raise MaterializationError("Set SCHEDULER_ENABLED=false for materialization runs.")
    if date_from and date_to and date_from > date_to:
        raise ValueError("date_from cannot be after date_to")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    league_id = int(CHL_LEAGUES[normalized]["id"])
    try:
        bundle = load_model_bundle_by_version(model_version, league_code=normalized)
    except ModelNotAvailableError as exc:
        raise MaterializationError(str(exc)) from exc

    resolved_model_version = str((bundle.get("_active_metadata") or {}).get("model_version") or bundle.get("model_version") or "")
    if resolved_model_version != model_version:
        raise MaterializationError(
            f"Requested model version {model_version} but bundle resolves to {resolved_model_version}."
        )
    model_family = str((bundle.get("_active_metadata") or {}).get("model_family") or bundle.get("model_family") or "")
    use_fast_v3 = model_family.startswith("whl_v3_")

    stmt = select(ChlGame).where(
        ChlGame.league_id == league_id,
        ChlGame.game_date.is_not(None),
        ChlGame.home_team_id.is_not(None),
        ChlGame.away_team_id.is_not(None),
        ChlGame.home_goal_count.is_not(None),
        ChlGame.away_goal_count.is_not(None),
    )
    if date_from:
        stmt = stmt.where(ChlGame.game_date >= date_from)
    if date_to:
        stmt = stmt.where(ChlGame.game_date <= date_to)
    games = db.scalars(stmt.order_by(ChlGame.game_date.asc(), ChlGame.game_id.asc())).all()

    teams = db.scalars(select(ChlTeam).where(ChlTeam.league_id == league_id)).all()
    teams_by_provider = {int(team.hockeytech_id): team for team in teams}
    context_by_game_id = _load_context_map(db, league_id) if use_fast_v3 else {}
    rolling = _load_rolling_snapshots(db, league_id) if use_fast_v3 else {}
    existing_ids = _existing_game_ids(db, league_id, model_version) if skip_existing and not dry_run else set()

    counts = Counter()
    for game in games:
        counts["games_scanned"] += 1
        if int(game.game_id) in existing_ids:
            counts["games_skipped_existing"] += 1
            continue

        home_team = teams_by_provider.get(int(game.home_team_id))
        away_team = teams_by_provider.get(int(game.away_team_id))
        if home_team is None or away_team is None:
            counts["games_skipped_missing_team"] += 1
            continue

        try:
            if use_fast_v3:
                payload = _fast_v3_payload(
                    game,
                    home_team_db_id=int(home_team.id),
                    away_team_db_id=int(away_team.id),
                    rolling=rolling,
                    context_by_game_id=context_by_game_id,
                )
            else:
                built = build_features_by_k(
                    db=db,
                    home_team_hockeytech_id=int(game.home_team_id),
                    away_team_hockeytech_id=int(game.away_team_id),
                    game_date=game.game_date,
                    league_code=normalized,
                )
                payload = {
                    "game_id": int(game.game_id),
                    "game_date": game.game_date.isoformat(),
                    "home_team_id": int(game.home_team_id),
                    "away_team_id": int(game.away_team_id),
                    "features_by_k": built["features_by_k"],
                    "context_features": built.get("context_features") or {},
                }
        except (InsufficientHistoryError, TeamNotFoundError, KeyError, ValueError) as exc:
            counts[f"games_skipped_{type(exc).__name__}"] += 1
            continue

        try:
            result = predict_from_payload(payload, bundle=bundle, league_code=normalized)
            if not dry_run:
                _persist_prediction_rows(
                    db,
                    game,
                    home_team,
                    away_team,
                    result,
                    commit=False,
                    league_code=normalized,
                )
        except Exception:
            db.rollback()
            raise

        counts["games_predicted"] += 1
        counts["rows_materialized"] += len(K_VALUES)
        if not dry_run and counts["games_predicted"] % batch_size == 0:
            db.commit()

    if not dry_run:
        db.commit()

    return {
        "league_code": normalized,
        "model_version": model_version,
        "model_family": model_family,
        "dry_run": dry_run,
        "fast_v3_path": use_fast_v3,
        **{key: int(value) for key, value in counts.items()},
    }
