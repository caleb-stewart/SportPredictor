from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import ChlGame, ChlModelCompareRun
from services.feature_builder import InsufficientHistoryError, TeamNotFoundError, build_features_by_k
from services.predictor import ModelNotAvailableError, PayloadContractError, load_model_bundle_by_version, predict_from_payload
from services.data_backend import (
    DataBackendError,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)


class ModelCompareError(RuntimeError):
    pass


class ModelCompareValidationError(ModelCompareError):
    pass


class ModelCompareNotFoundError(ModelCompareError):
    pass


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _completed_game_filters(league_id: int) -> list[Any]:
    return [
        ChlGame.league_id == league_id,
        ChlGame.game_date.is_not(None),
        ChlGame.home_goal_count.is_not(None),
        ChlGame.away_goal_count.is_not(None),
    ]


def _resolve_date_bounds(db: Session, league_id: int, date_from: dt.date | None, date_to: dt.date | None) -> tuple[dt.date, dt.date]:
    min_date, max_date = db.execute(
        select(func.min(ChlGame.game_date), func.max(ChlGame.game_date)).where(*_completed_game_filters(league_id))
    ).one()
    if min_date is None or max_date is None:
        raise ModelCompareValidationError("No completed games found in chl_games.")

    resolved_from = date_from or min_date
    resolved_to = date_to or max_date
    if resolved_from > resolved_to:
        raise ModelCompareValidationError("date_from cannot be after date_to.")
    return resolved_from, resolved_to


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int)
    p = np.clip(probs.astype(float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(np.mean(pred == y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _mcnemar_p_value(old_correct: np.ndarray, new_correct: np.ndarray) -> dict[str, Any]:
    b = int(np.sum((old_correct == 1) & (new_correct == 0)))
    c = int(np.sum((old_correct == 0) & (new_correct == 1)))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": 0, "p_value": 1.0, "method": "degenerate"}

    chi2_cc = ((abs(b - c) - 1.0) ** 2) / float(n)
    p_value = float(math.erfc(math.sqrt(max(chi2_cc, 0.0) / 2.0)))
    return {"b": b, "c": c, "n": n, "p_value": min(1.0, p_value), "method": "chi_square_cc"}


def _bootstrap_ci(old_correct: np.ndarray, new_correct: np.ndarray, samples: int = 2000, seed: int = 42) -> dict[str, Any]:
    n = len(old_correct)
    if n == 0:
        return {"samples": samples, "seed": seed, "ci95_low": None, "ci95_high": None, "delta_mean": 0.0}

    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(new_correct[idx]) - np.mean(old_correct[idx]))
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975]).tolist()
    return {
        "samples": samples,
        "seed": seed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "delta_mean": float(np.mean(deltas)),
    }


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "paired_games": 0,
            "overall": None,
            "by_season": [],
            "stat_tests": {
                "mcnemar": {"b": 0, "c": 0, "n": 0, "p_value": 1.0, "method": "degenerate"},
                "bootstrap_accuracy_delta": {
                    "samples": 2000,
                    "seed": 42,
                    "ci95_low": None,
                    "ci95_high": None,
                    "delta_mean": 0.0,
                },
            },
            "pass_criteria": {
                "accuracy_improved": False,
                "log_loss_improved": False,
                "mcnemar_significant": False,
                "accuracy_delta_ci_above_zero": False,
            },
            "proved_better": False,
        }

    y = np.array([int(r["actual_home_win"]) for r in rows], dtype=int)
    old_probs = np.array([float(r["baseline_home_prob"]) for r in rows], dtype=float)
    new_probs = np.array([float(r["candidate_home_prob"]) for r in rows], dtype=float)

    old_metrics = _binary_metrics(y, old_probs)
    new_metrics = _binary_metrics(y, new_probs)

    old_correct = ((old_probs >= 0.5).astype(int) == y).astype(int)
    new_correct = ((new_probs >= 0.5).astype(int) == y).astype(int)

    mcnemar = _mcnemar_p_value(old_correct, new_correct)
    boot = _bootstrap_ci(old_correct, new_correct)

    by_season: list[dict[str, Any]] = []
    seasons = sorted({str(r.get("season_name") or "unknown") for r in rows})
    for season in seasons:
        s_rows = [r for r in rows if str(r.get("season_name") or "unknown") == season]
        if len(s_rows) < 100:
            continue
        y_s = np.array([int(r["actual_home_win"]) for r in s_rows], dtype=int)
        old_s = np.array([float(r["baseline_home_prob"]) for r in s_rows], dtype=float)
        new_s = np.array([float(r["candidate_home_prob"]) for r in s_rows], dtype=float)
        old_m = _binary_metrics(y_s, old_s)
        new_m = _binary_metrics(y_s, new_s)
        by_season.append(
            {
                "season_name": season,
                "games": len(s_rows),
                "baseline": old_m,
                "candidate": new_m,
                "delta_new_minus_old": {
                    "accuracy": float(new_m["accuracy"] - old_m["accuracy"]),
                    "log_loss": float(new_m["log_loss"] - old_m["log_loss"]),
                    "brier": float(new_m["brier"] - old_m["brier"]),
                },
            }
        )

    pass_criteria = {
        "accuracy_improved": bool(new_metrics["accuracy"] > old_metrics["accuracy"]),
        "log_loss_improved": bool(new_metrics["log_loss"] < old_metrics["log_loss"]),
        "mcnemar_significant": bool(float(mcnemar["p_value"]) < 0.05),
        "accuracy_delta_ci_above_zero": bool(boot["ci95_low"] is not None and float(boot["ci95_low"]) > 0.0),
    }

    return {
        "paired_games": len(rows),
        "overall": {
            "baseline": old_metrics,
            "candidate": new_metrics,
            "delta_new_minus_old": {
                "accuracy": float(new_metrics["accuracy"] - old_metrics["accuracy"]),
                "log_loss": float(new_metrics["log_loss"] - old_metrics["log_loss"]),
                "brier": float(new_metrics["brier"] - old_metrics["brier"]),
            },
        },
        "by_season": by_season,
        "stat_tests": {"mcnemar": mcnemar, "bootstrap_accuracy_delta": boot},
        "pass_criteria": pass_criteria,
        "proved_better": all(pass_criteria.values()),
    }


def _write_report_file(run_id: uuid.UUID, payload: dict[str, Any]) -> str:
    root = Path(__file__).resolve().parents[1] / "reports" / "compare"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def run_model_compare(
    db: Session,
    *,
    candidate_model_version: str,
    baseline_model_version: str,
    date_from: dt.date | None,
    date_to: dt.date | None,
    mode: str = "frozen_replay",
    league_code: str | None = None,
) -> dict[str, Any]:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise ModelCompareValidationError(str(exc)) from exc
    store = primary_store()
    league_id = resolve_league_id_for_store(db, store, normalized_league)
    if league_id is None:
        raise ModelCompareValidationError(f"Unable to resolve league scope for league_code={normalized_league}")

    if mode != "frozen_replay":
        raise ModelCompareValidationError("Only mode='frozen_replay' is supported.")
    if candidate_model_version == baseline_model_version:
        raise ModelCompareValidationError("candidate_model_version and baseline_model_version must differ.")

    resolved_from, resolved_to = _resolve_date_bounds(db, league_id, date_from, date_to)
    candidate_bundle = load_model_bundle_by_version(candidate_model_version, league_code=normalized_league)
    baseline_bundle = load_model_bundle_by_version(baseline_model_version, league_code=normalized_league)

    run_row = ChlModelCompareRun(
        id=uuid.uuid4(),
        league_id=league_id,
        status="running",
        mode=mode,
        baseline_model_version=baseline_model_version,
        candidate_model_version=candidate_model_version,
        date_from=resolved_from,
        date_to=resolved_to,
        games_scanned=0,
        games_compared=0,
    )
    db.add(run_row)
    db.commit()

    try:
        games = db.scalars(
            select(ChlGame)
            .where(*_completed_game_filters(league_id))
            .where(ChlGame.game_date >= resolved_from, ChlGame.game_date <= resolved_to)
            .order_by(ChlGame.game_date.asc(), ChlGame.game_id.asc())
        ).all()

        compared_rows: list[dict[str, Any]] = []
        scanned = len(games)
        for game in games:
            if game.game_date is None or game.home_team_id is None or game.away_team_id is None:
                continue
            try:
                built = build_features_by_k(
                    db=db,
                    home_team_hockeytech_id=game.home_team_id,
                    away_team_hockeytech_id=game.away_team_id,
                    game_date=game.game_date,
                    league_code=normalized_league,
                )
            except (InsufficientHistoryError, TeamNotFoundError):
                continue

            payload = {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
                "features_by_k": built["features_by_k"],
                "context_features": built.get("context_features") or {},
            }
            try:
                baseline = predict_from_payload(payload, bundle=baseline_bundle, league_code=normalized_league)
                candidate = predict_from_payload(payload, bundle=candidate_bundle, league_code=normalized_league)
            except (PayloadContractError, ModelNotAvailableError):
                continue

            compared_rows.append(
                {
                    "game_id": game.game_id,
                    "game_date": str(game.game_date),
                    "season_name": game.season_name,
                    "actual_home_win": 1 if int(game.home_goal_count) > int(game.away_goal_count) else 0,
                    "baseline_home_prob": float(baseline["home_team_prob"]),
                    "candidate_home_prob": float(candidate["home_team_prob"]),
                }
            )

        summary = _build_summary(compared_rows)
        payload = {
            "run_id": str(run_row.id),
            "generated_at_utc": _iso_utc_now(),
            "mode": mode,
            "baseline_model_version": baseline_model_version,
            "candidate_model_version": candidate_model_version,
            "date_from": str(resolved_from),
            "date_to": str(resolved_to),
            "counts": {"games_scanned": scanned, "games_compared": len(compared_rows)},
            "proof_summary": summary,
            "proved_better": bool(summary.get("proved_better")),
        }
        report_path = _write_report_file(run_row.id, payload)
        payload["report_path"] = report_path

        run_row.status = "completed"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.games_scanned = scanned
        run_row.games_compared = len(compared_rows)
        run_row.proof_json = payload
        db.commit()
        db.refresh(run_row)

        return {
            "run_id": str(run_row.id),
            "status": run_row.status,
            "mode": run_row.mode,
            "baseline_model_version": run_row.baseline_model_version,
            "candidate_model_version": run_row.candidate_model_version,
            "games_scanned": run_row.games_scanned,
            "games_compared": run_row.games_compared,
            "proof_summary": summary,
        }
    except Exception as exc:  # noqa: BLE001
        run_row.status = "failed"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.error_text = str(exc)
        db.commit()
        raise


def get_model_compare_report(db: Session, run_id: str, league_code: str | None = None) -> dict[str, Any]:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise ModelCompareValidationError(str(exc)) from exc
    store = primary_store()
    league_id = resolve_league_id_for_store(db, store, normalized_league)
    if league_id is None:
        raise ModelCompareValidationError(f"Unable to resolve league scope for league_code={normalized_league}")

    try:
        parsed = uuid.UUID(run_id)
    except ValueError as exc:
        raise ModelCompareValidationError("run_id must be a valid UUID.") from exc

    run = db.scalar(
        select(ChlModelCompareRun).where(ChlModelCompareRun.id == parsed, ChlModelCompareRun.league_id == league_id)
    )
    if run is None:
        raise ModelCompareNotFoundError(f"Model compare run not found: {run_id}")

    proof = run.proof_json or {}
    summary = proof.get("proof_summary") if isinstance(proof, dict) else None
    return {
        "run_id": str(run.id),
        "status": run.status,
        "mode": run.mode,
        "baseline_model_version": run.baseline_model_version,
        "candidate_model_version": run.candidate_model_version,
        "date_from": run.date_from,
        "date_to": run.date_to,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "games_scanned": run.games_scanned,
        "games_compared": run.games_compared,
        "proof_summary": summary,
        "error_text": run.error_text,
    }
