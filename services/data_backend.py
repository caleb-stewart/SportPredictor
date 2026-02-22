from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from core.config import settings
from db.models import (
    ChlCustomPredictionRecord,
    ChlExperiment,
    ChlFeatureRegistry,
    ChlGame,
    ChlLeague,
    ChlModelCompareRun,
    ChlPredictionRecord,
    ChlPredictionRecordArchive,
    ChlReplayRun,
    ChlRollingAverage,
    ChlTeam,
)


class DataBackendError(RuntimeError):
    pass


CHL_LEAGUES: dict[str, dict[str, Any]] = {
    "whl": {
        "id": 1,
        "code": "whl",
        "name": "Western Hockey League",
        "sport": "hockey",
        "provider": "hockeytech",
        "provider_league_code": "whl",
        "timezone": "America/Los_Angeles",
        "active": True,
    },
    "ohl": {
        "id": 2,
        "code": "ohl",
        "name": "Ontario Hockey League",
        "sport": "hockey",
        "provider": "hockeytech",
        "provider_league_code": "ohl",
        "timezone": "America/Toronto",
        "active": True,
    },
    "lhjmq": {
        "id": 3,
        "code": "lhjmq",
        "name": "Quebec Maritimes Junior Hockey League",
        "sport": "hockey",
        "provider": "hockeytech",
        "provider_league_code": "lhjmq",
        "timezone": "America/Halifax",
        "active": True,
    },
}


@dataclass(frozen=True)
class HockeyStoreModels:
    name: str
    league_scoped: bool
    league_model: type | None
    team_model: type
    game_model: type
    rolling_model: type
    prediction_model: type
    custom_prediction_model: type
    archive_model: type
    replay_run_model: type
    model_compare_run_model: type
    experiment_model: type
    feature_registry_model: type
    rolling_team_id_field: str


CHL_STORE = HockeyStoreModels(
    name="chl",
    league_scoped=True,
    league_model=ChlLeague,
    team_model=ChlTeam,
    game_model=ChlGame,
    rolling_model=ChlRollingAverage,
    prediction_model=ChlPredictionRecord,
    custom_prediction_model=ChlCustomPredictionRecord,
    archive_model=ChlPredictionRecordArchive,
    replay_run_model=ChlReplayRun,
    model_compare_run_model=ChlModelCompareRun,
    experiment_model=ChlExperiment,
    feature_registry_model=ChlFeatureRegistry,
    rolling_team_id_field="team_id",
)


def normalize_league_code(league_code: str | None) -> str:
    normalized = (league_code or settings.default_league_code).strip().lower()
    if normalized == "qmjhl":
        return "lhjmq"
    return normalized


def require_supported_league_code(league_code: str | None) -> str:
    normalized = normalize_league_code(league_code)
    if normalized not in CHL_LEAGUES:
        raise DataBackendError(f"Unsupported league_code: {league_code}")
    return normalized


def hockeytech_client_code_for_league(league_code: str | None) -> str:
    normalized = require_supported_league_code(league_code)
    return str(CHL_LEAGUES[normalized]["provider_league_code"])


def hockeytech_api_key_for_league(league_code: str | None) -> str:
    normalized = require_supported_league_code(league_code)
    key_by_league = {
        "whl": settings.hockeytech_api_key_whl,
        "ohl": settings.hockeytech_api_key_ohl,
        "lhjmq": settings.hockeytech_api_key_lhjmq,
    }
    candidate = (key_by_league.get(normalized) or "").strip()
    if candidate:
        return candidate

    fallback = (settings.hockeytech_api_key or "").strip()
    if fallback:
        return fallback

    raise DataBackendError(
        "Missing HockeyTech API key. Set HOCKEYTECH_API_KEY_<LEAGUE> or fallback HOCKEYTECH_API."
    )


def primary_store() -> HockeyStoreModels:
    return CHL_STORE


def secondary_store() -> HockeyStoreModels | None:
    return None


def build_store_chain() -> list[HockeyStoreModels]:
    primary = primary_store()
    secondary = secondary_store()
    if secondary is None:
        return [primary]
    return [primary, secondary]


def apply_league_scope(stmt, model: type, league_id: int | None):
    if hasattr(model, "league_id"):
        if league_id is None:
            raise DataBackendError("league_id is required for league-scoped tables")
        return stmt.where(getattr(model, "league_id") == league_id)
    return stmt


def game_conflict_columns(store: HockeyStoreModels) -> list[str]:
    return ["league_id", "game_id"] if store.league_scoped else ["game_id"]


def rolling_conflict_columns(store: HockeyStoreModels) -> list[str]:
    if store.league_scoped:
        return ["league_id", "game_id", "k_value", store.rolling_team_id_field]
    return ["game_id", "k_value", store.rolling_team_id_field]


def prediction_conflict_columns(store: HockeyStoreModels) -> list[str]:
    return ["league_id", "game_id", "k_value"] if store.league_scoped else ["game_id", "k_value"]


def ensure_chl_league(db: Session, league_code: str | None) -> ChlLeague:
    normalized = normalize_league_code(league_code)
    payload = CHL_LEAGUES.get(normalized)
    if payload is None:
        raise DataBackendError(f"Unsupported league_code: {league_code}")

    row = db.scalar(select(ChlLeague).where(ChlLeague.code == normalized))
    if row is not None:
        return row

    db.execute(
        insert(ChlLeague)
        .values(**payload)
        .on_conflict_do_update(
            index_elements=[ChlLeague.code],
            set_={
                "name": payload["name"],
                "sport": payload["sport"],
                "provider": payload["provider"],
                "provider_league_code": payload["provider_league_code"],
                "timezone": payload["timezone"],
                "active": payload["active"],
            },
        )
    )
    db.flush()
    row = db.scalar(select(ChlLeague).where(ChlLeague.code == normalized))
    if row is None:
        raise DataBackendError(f"Failed to resolve CHL league row for code={normalized}")
    return row


def resolve_league_id_for_store(db: Session, store: HockeyStoreModels, league_code: str | None) -> int | None:
    if not store.league_scoped:
        return None
    normalized = require_supported_league_code(league_code)
    if not hasattr(db, "scalar"):
        payload = CHL_LEAGUES.get(normalized)
        if payload is None:
            raise DataBackendError(f"Unsupported league_code: {league_code}")
        return int(payload["id"])
    return int(ensure_chl_league(db, normalized).id)
