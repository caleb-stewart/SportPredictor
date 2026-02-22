from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url

from core.config import settings
from services.trainer import TrainerConfig, train_model_package
from services.training_dataset import DbConfig


class TrainingError(RuntimeError):
    pass


def _json_atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_store_root_from_output(output_dir: str | None) -> Path | None:
    if not output_dir:
        return None
    out = Path(output_dir).resolve()
    if out.exists() and out.is_dir():
        return out.parent
    return None


def _pending_pointer_path_for(details: dict[str, Any]) -> Path | None:
    root = _resolve_store_root_from_output(details.get("output_dir"))
    if root is None:
        return None
    return root / "pending_model.json"


def _stage_pending_candidate(details: dict[str, Any]) -> None:
    gates = details.get("gates") or {}
    if not bool(gates.get("all_passed")):
        return

    version = details.get("version")
    output_dir = details.get("output_dir")
    if not version or not output_dir:
        return

    output_path = Path(output_dir).resolve()
    metadata_path = output_path / "metadata.json"
    bundle_path = output_path / "model_bundle.joblib"
    metrics_path = output_path / "metrics.json"

    if not metadata_path.exists() or not bundle_path.exists() or not metrics_path.exists():
        return

    pending_path = _pending_pointer_path_for(details)
    if pending_path is None:
        return

    pending_payload = {
        "model_version": version,
        "model_family": details.get("model_family") or "whl_v3_hybrid_branch_stacker",
        "metadata_path": str(metadata_path),
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "staged_at_utc": _iso_utc_now(),
    }
    _json_atomic_write(pending_path, pending_payload)


def _promote_from_pending_if_present(details: dict[str, Any]) -> dict[str, Any]:
    pending_path = _pending_pointer_path_for(details)
    if pending_path is None or not pending_path.exists():
        details["promoted"] = False
        details["promotion"] = {"ok": False, "reason": "pending candidate not found"}
        return details

    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    metadata_path = Path(str(pending.get("metadata_path") or "")).resolve()
    if not metadata_path.exists():
        details["promoted"] = False
        details["promotion"] = {"ok": False, "reason": f"metadata missing: {metadata_path}"}
        return details

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gates_passed = bool(metadata.get("gates_passed"))
    if not gates_passed:
        details["promoted"] = False
        details["promotion"] = {"ok": False, "reason": "gates did not pass"}
        return details

    active_path = pending_path.parent / "active_model.json"
    _json_atomic_write(active_path, metadata)
    try:
        pending_path.unlink(missing_ok=True)
    except OSError:
        pass

    details["promoted"] = True
    details["promotion"] = {
        "ok": True,
        "active_model_path": str(active_path),
        "model_version": metadata.get("model_version"),
    }
    return details


def _league_model_store_root(league_code: str) -> Path:
    root = Path(settings.model_store_root).resolve()
    return (root / league_code).resolve()


def _run_training(promote: bool, league_code: str) -> dict[str, Any]:
    db_url = make_url(settings.database_url)
    db = DbConfig(
        host=db_url.host or "localhost",
        port=str(db_url.port or 5432),
        dbname=db_url.database or "sportpredictor_development",
        user=db_url.username or "postgres",
        password=db_url.password or "",
    )

    config = TrainerConfig(
        output_root=_league_model_store_root(league_code),
        model_version=None,
        no_promote=not promote,
        export_dataset_path=None,
        min_accuracy=settings.min_accuracy_gate,
        min_season_accuracy=settings.min_season_accuracy_gate,
        min_season_games=settings.min_season_games_gate,
        base_oof_splits=5,
        meta_cv_splits=5,
        min_train_size=600,
        db=db,
        league_code=league_code,
    )

    try:
        details, return_code = train_model_package(config)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Training failed: {exc}") from exc

    details["return_code"] = return_code
    details["stderr"] = ""
    if return_code == 2:
        details["promoted"] = False
        details["ok"] = False
    else:
        details["ok"] = True

    return details


def train_and_maybe_promote(promote: bool = True, league_code: str = "whl") -> dict[str, Any]:
    details = _run_training(promote=False, league_code=league_code)
    _stage_pending_candidate(details)
    details["league_code"] = league_code

    if promote and bool(details.get("ok")) and bool((details.get("gates") or {}).get("all_passed")):
        details = _promote_from_pending_if_present(details)
    elif not promote:
        details["promoted"] = False
        details["promotion"] = {"ok": False, "reason": "staged only; promote not requested"}

    return details


def promote_staged_model(league_code: str = "whl") -> dict[str, Any]:
    root = _league_model_store_root(league_code)
    pending_path = root / "pending_model.json"
    if not pending_path.exists():
        return {"ok": False, "promoted": False, "reason": "no staged model found"}

    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    metadata_path = Path(str(pending.get("metadata_path") or "")).resolve()
    if not metadata_path.exists():
        return {"ok": False, "promoted": False, "reason": f"metadata missing: {metadata_path}"}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not bool(metadata.get("gates_passed")):
        return {"ok": False, "promoted": False, "reason": "staged model did not pass gates"}

    active_path = root / "active_model.json"
    _json_atomic_write(active_path, metadata)
    try:
        pending_path.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "ok": True,
        "promoted": True,
        "active_model_path": str(active_path),
        "model_version": metadata.get("model_version"),
        "model_family": metadata.get("model_family"),
    }
