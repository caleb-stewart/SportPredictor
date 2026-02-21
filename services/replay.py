from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sqlalchemy import and_, delete, func, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from core.config import settings
from db.models import (
    WhlGame,
    WhlPredictionRecord,
    WhlPredictionRecordArchive,
    WhlReplayRun,
)
from services.feature_builder import (
    InsufficientHistoryError,
    TeamNotFoundError,
    build_features_by_k,
)
from services.prediction_pipeline import K_VALUES, _persist_prediction_rows
from services.predictor import (
    ModelNotAvailableError,
    PayloadContractError,
    is_active_model_pointer_unchanged,
    load_frozen_active_model,
    predict_from_payload,
)


class ReplayError(RuntimeError):
    pass


class ReplayValidationError(ReplayError):
    pass


class ReplayNotFoundError(ReplayError):
    pass


class ReplaySafetyError(ReplayError):
    pass


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _completed_game_filters() -> list[Any]:
    return [
        WhlGame.game_date.is_not(None),
        WhlGame.home_goal_count.is_not(None),
        WhlGame.away_goal_count.is_not(None),
    ]


def completed_games_stmt(date_from: dt.date | None = None, date_to: dt.date | None = None):
    stmt = select(WhlGame).where(*_completed_game_filters())
    if date_from is not None:
        stmt = stmt.where(WhlGame.game_date >= date_from)
    if date_to is not None:
        stmt = stmt.where(WhlGame.game_date <= date_to)
    return stmt.order_by(WhlGame.game_date.asc(), WhlGame.game_id.asc())


def _prediction_rows_stmt(date_from: dt.date, date_to: dt.date):
    return (
        select(WhlPredictionRecord)
        .join(WhlGame, WhlPredictionRecord.game_id == WhlGame.game_id)
        .where(*_completed_game_filters())
        .where(WhlGame.game_date >= date_from, WhlGame.game_date <= date_to)
        .where(WhlPredictionRecord.k_value.in_(K_VALUES))
    )


def _resolve_date_bounds(
    db: Session,
    date_from: dt.date | None,
    date_to: dt.date | None,
) -> tuple[dt.date, dt.date]:
    min_date, max_date = db.execute(
        select(func.min(WhlGame.game_date), func.max(WhlGame.game_date)).where(*_completed_game_filters())
    ).one()

    if min_date is None or max_date is None:
        raise ReplayValidationError("No completed games found in whl_games.")

    resolved_from = date_from or min_date
    resolved_to = date_to or max_date
    if resolved_from > resolved_to:
        raise ReplayValidationError("date_from cannot be after date_to.")
    return resolved_from, resolved_to


def _chunked(seq: list[Any], size: int = 1000) -> Iterable[list[Any]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _archive_rows(
    db: Session,
    old_rows: list[WhlPredictionRecord],
    run_id: uuid.UUID,
    archive_label: str | None,
) -> int:
    if not old_rows:
        return 0

    now = dt.datetime.now(dt.UTC)
    values = [
        {
            "id": row.id,
            "game_id": row.game_id,
            "k_value": row.k_value,
            "home_team_id": row.home_team_id,
            "away_team_id": row.away_team_id,
            "predicted_winner_id": row.predicted_winner_id,
            "home_team_probability": row.home_team_probability,
            "away_team_probability": row.away_team_probability,
            "actual_winner_id": row.actual_winner_id,
            "correct": row.correct,
            "prediction_date": row.prediction_date,
            "model_version": row.model_version,
            "model_family": row.model_family,
            "raw_model_outputs": row.raw_model_outputs,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "archive_run_id": run_id,
            "archive_label": archive_label,
            "archived_at": now,
        }
        for row in old_rows
    ]

    for chunk in _chunked(values):
        db.execute(insert(WhlPredictionRecordArchive), chunk)
    db.commit()
    return len(values)


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


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

    if n > 400:
        chi2_cc = ((abs(b - c) - 1.0) ** 2) / float(n)
        p_value = float(math.erfc(math.sqrt(max(chi2_cc, 0.0) / 2.0)))
        return {"b": b, "c": c, "n": n, "p_value": min(1.0, p_value), "method": "chi_square_cc"}

    x = min(b, c)
    log_two = math.log(2.0)
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - (n * log_two)
        for i in range(x + 1)
    ]
    max_log = max(terms)
    cdf = math.exp(max_log) * sum(math.exp(t - max_log) for t in terms)
    return {"b": b, "c": c, "n": n, "p_value": min(1.0, 2.0 * cdf), "method": "exact_binomial"}


def _bootstrap_accuracy_delta_ci(
    old_correct: np.ndarray,
    new_correct: np.ndarray,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
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


def build_proof_summary(
    paired_rows: list[dict[str, Any]],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if not paired_rows:
        return {
            "paired_rows": 0,
            "overall": None,
            "by_k": {},
            "stat_tests": {
                "mcnemar": {"b": 0, "c": 0, "n": 0, "p_value": 1.0, "method": "degenerate"},
                "bootstrap_accuracy_delta": {
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed,
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

    y = np.array([int(row["actual_home_win"]) for row in paired_rows], dtype=int)
    old_probs = np.array([_safe_float(row["old_home_prob"]) for row in paired_rows], dtype=float)
    new_probs = np.array([_safe_float(row["new_home_prob"]) for row in paired_rows], dtype=float)

    old_metrics = _binary_metrics(y, old_probs)
    new_metrics = _binary_metrics(y, new_probs)

    old_correct = ((old_probs >= 0.5).astype(int) == y).astype(int)
    new_correct = ((new_probs >= 0.5).astype(int) == y).astype(int)

    mcnemar = _mcnemar_p_value(old_correct=old_correct, new_correct=new_correct)
    boot = _bootstrap_accuracy_delta_ci(
        old_correct=old_correct,
        new_correct=new_correct,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    by_k: dict[str, Any] = {}
    for k in K_VALUES:
        rows = [row for row in paired_rows if int(row["k_value"]) == int(k)]
        if not rows:
            by_k[str(k)] = None
            continue

        y_k = np.array([int(row["actual_home_win"]) for row in rows], dtype=int)
        old_k = np.array([_safe_float(row["old_home_prob"]) for row in rows], dtype=float)
        new_k = np.array([_safe_float(row["new_home_prob"]) for row in rows], dtype=float)
        old_k_metrics = _binary_metrics(y_k, old_k)
        new_k_metrics = _binary_metrics(y_k, new_k)
        by_k[str(k)] = {
            "rows": len(rows),
            "old": old_k_metrics,
            "new": new_k_metrics,
            "delta_new_minus_old": {
                "accuracy": float(new_k_metrics["accuracy"] - old_k_metrics["accuracy"]),
                "log_loss": float(new_k_metrics["log_loss"] - old_k_metrics["log_loss"]),
                "brier": float(new_k_metrics["brier"] - old_k_metrics["brier"]),
            },
        }

    delta_accuracy = float(new_metrics["accuracy"] - old_metrics["accuracy"])
    delta_log_loss = float(new_metrics["log_loss"] - old_metrics["log_loss"])
    delta_brier = float(new_metrics["brier"] - old_metrics["brier"])

    pass_criteria = {
        "accuracy_improved": bool(new_metrics["accuracy"] > old_metrics["accuracy"]),
        "log_loss_improved": bool(new_metrics["log_loss"] < old_metrics["log_loss"]),
        "mcnemar_significant": bool(float(mcnemar["p_value"]) < 0.05),
        "accuracy_delta_ci_above_zero": bool(
            boot["ci95_low"] is not None and float(boot["ci95_low"]) > 0.0
        ),
    }

    return {
        "paired_rows": len(paired_rows),
        "overall": {
            "old": old_metrics,
            "new": new_metrics,
            "delta_new_minus_old": {
                "accuracy": delta_accuracy,
                "log_loss": delta_log_loss,
                "brier": delta_brier,
            },
        },
        "by_k": by_k,
        "stat_tests": {
            "mcnemar": mcnemar,
            "bootstrap_accuracy_delta": boot,
        },
        "pass_criteria": pass_criteria,
        "proved_better": all(pass_criteria.values()),
    }


def _write_report_file(run_id: uuid.UUID, report_payload: dict[str, Any]) -> str:
    reports_root = Path(__file__).resolve().parents[1] / "reports" / "replay"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"{run_id}.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(report_path)


def _rollback_replayed_rows(
    db: Session,
    run_id: uuid.UUID,
    replayed_keys: set[tuple[int, int]],
) -> None:
    if replayed_keys:
        key_list = list(replayed_keys)
        for chunk in _chunked(key_list):
            db.execute(
                delete(WhlPredictionRecord).where(
                    tuple_(WhlPredictionRecord.game_id, WhlPredictionRecord.k_value).in_(chunk)
                )
            )

    archived_rows = db.scalars(
        select(WhlPredictionRecordArchive).where(WhlPredictionRecordArchive.archive_run_id == run_id)
    ).all()
    if archived_rows:
        restore_values = [
            {
                "id": row.id,
                "game_id": row.game_id,
                "k_value": row.k_value,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "predicted_winner_id": row.predicted_winner_id,
                "home_team_probability": row.home_team_probability,
                "away_team_probability": row.away_team_probability,
                "actual_winner_id": row.actual_winner_id,
                "correct": row.correct,
                "prediction_date": row.prediction_date,
                "model_version": row.model_version,
                "model_family": row.model_family,
                "raw_model_outputs": row.raw_model_outputs,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in archived_rows
        ]
        for chunk in _chunked(restore_values):
            db.execute(insert(WhlPredictionRecord), chunk)

    db.commit()


def _extract_component_prob(result: dict[str, Any], k_value: int) -> tuple[float, float]:
    comp = (result.get("k_components") or {}).get(str(k_value)) or {}
    home_prob = _safe_float(comp.get("home_team_prob"), default=_safe_float(result.get("home_team_prob"), 0.5))
    away_prob = _safe_float(comp.get("away_team_prob"), default=1.0 - home_prob)
    return home_prob, away_prob


def _build_pairs_for_proof(
    games_by_id: dict[int, WhlGame],
    old_rows_by_key: dict[tuple[int, int], dict[str, Any]],
    new_rows_by_key: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for key, new_row in new_rows_by_key.items():
        old_row = old_rows_by_key.get(key)
        if old_row is None:
            continue
        old_home_prob = old_row.get("home_team_probability")
        if old_home_prob is None:
            continue

        game = games_by_id.get(new_row["game_id"])
        if game is None or game.home_goal_count is None or game.away_goal_count is None:
            continue

        paired.append(
            {
                "game_id": new_row["game_id"],
                "k_value": new_row["k_value"],
                "actual_home_win": 1 if int(game.home_goal_count) > int(game.away_goal_count) else 0,
                "old_home_prob": _safe_float(old_home_prob),
                "new_home_prob": _safe_float(new_row["home_team_probability"]),
            }
        )
    return paired


def _new_row_stub(game_id: int, k_value: int, result: dict[str, Any]) -> dict[str, Any]:
    home_prob, away_prob = _extract_component_prob(result=result, k_value=k_value)
    return {
        "game_id": game_id,
        "k_value": k_value,
        "home_team_probability": home_prob,
        "away_team_probability": away_prob,
        "model_version": result.get("model_version"),
    }


def run_frozen_model_replay(
    db: Session,
    *,
    date_from: dt.date | None,
    date_to: dt.date | None,
    dry_run: bool,
    overwrite: bool,
    archive_label: str | None,
) -> dict[str, Any]:
    if not dry_run and not overwrite:
        raise ReplayValidationError("overwrite must be true when dry_run is false.")
    if not dry_run and settings.scheduler_enabled:
        raise ReplayValidationError("Set SCHEDULER_ENABLED=false for replay runs.")

    frozen_bundle, frozen_pointer = load_frozen_active_model()
    frozen_model_version = frozen_pointer.get("model_version")

    resolved_from, resolved_to = _resolve_date_bounds(db=db, date_from=date_from, date_to=date_to)

    run_id = uuid.uuid4()
    run_row = WhlReplayRun(
        id=run_id,
        status="running",
        started_at=dt.datetime.now(dt.UTC),
        date_from=resolved_from,
        date_to=resolved_to,
        active_model_version=frozen_model_version,
        games_scanned=0,
        games_predicted=0,
        games_skipped=0,
        rows_upserted=0,
    )
    db.add(run_row)
    db.commit()

    skip_reasons: dict[str, int] = {}
    replayed_keys: set[tuple[int, int]] = set()
    new_rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    try:
        games = db.scalars(completed_games_stmt(resolved_from, resolved_to)).all()
        games_by_id = {game.game_id: game for game in games}

        old_rows = db.scalars(_prediction_rows_stmt(resolved_from, resolved_to)).all()
        old_rows_by_key = {
            (row.game_id, row.k_value): {
                "home_team_probability": (
                    None if row.home_team_probability is None else float(row.home_team_probability)
                ),
                "model_version": row.model_version,
            }
            for row in old_rows
        }

        if not dry_run:
            _archive_rows(db=db, old_rows=old_rows, run_id=run_id, archive_label=archive_label)

        games_predicted = 0
        games_skipped = 0
        rows_upserted = 0

        for game in games:
            if not is_active_model_pointer_unchanged(frozen_pointer):
                raise ReplaySafetyError("Active model pointer changed during replay run.")

            if game.game_date is None:
                games_skipped += 1
                skip_reasons["missing_game_date"] = skip_reasons.get("missing_game_date", 0) + 1
                continue
            if game.home_team_id is None or game.away_team_id is None:
                games_skipped += 1
                skip_reasons["missing_team_ids"] = skip_reasons.get("missing_team_ids", 0) + 1
                continue

            try:
                built = build_features_by_k(
                    db=db,
                    home_team_hockeytech_id=game.home_team_id,
                    away_team_hockeytech_id=game.away_team_id,
                    game_date=game.game_date,
                )
            except InsufficientHistoryError:
                games_skipped += 1
                skip_reasons["insufficient_history"] = skip_reasons.get("insufficient_history", 0) + 1
                continue
            except TeamNotFoundError:
                games_skipped += 1
                skip_reasons["team_not_found"] = skip_reasons.get("team_not_found", 0) + 1
                continue

            payload = {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
                "features_by_k": built["features_by_k"],
            }

            try:
                result = predict_from_payload(payload, bundle=frozen_bundle)
            except (ModelNotAvailableError, PayloadContractError):
                games_skipped += 1
                skip_reasons["prediction_error"] = skip_reasons.get("prediction_error", 0) + 1
                continue

            games_predicted += 1

            replayed_at_utc = _iso_utc_now()
            metadata_by_k: dict[int, dict[str, Any]] = {}
            for k in K_VALUES:
                key = (game.game_id, k)
                replayed_keys.add(key)
                old_model_version = (old_rows_by_key.get(key) or {}).get("model_version")
                metadata_by_k[k] = {
                    "replay_run_id": str(run_id),
                    "replay_mode": "frozen_active",
                    "replayed_at_utc": replayed_at_utc,
                    "old_model_version": old_model_version,
                    "new_model_version": result.get("model_version"),
                }
                new_rows_by_key[key] = _new_row_stub(game_id=game.game_id, k_value=k, result=result)

            if not dry_run:
                rows_upserted += _persist_prediction_rows(
                    db=db,
                    game=game,
                    home_team=built["home_team"],
                    away_team=built["away_team"],
                    result=result,
                    extra_raw_model_outputs_by_k=metadata_by_k,
                    commit=True,
                )

        paired_rows = _build_pairs_for_proof(
            games_by_id=games_by_id,
            old_rows_by_key=old_rows_by_key,
            new_rows_by_key=new_rows_by_key,
        )
        proof_summary = build_proof_summary(paired_rows=paired_rows)

        status = "completed"
        if not dry_run and not proof_summary.get("proved_better", False):
            _rollback_replayed_rows(db=db, run_id=run_id, replayed_keys=replayed_keys)
            status = "rolled_back"

        report_payload = {
            "run_id": str(run_id),
            "status": status,
            "replay_mode": "frozen_active",
            "generated_at_utc": _iso_utc_now(),
            "active_model_version": frozen_model_version,
            "date_from": str(resolved_from),
            "date_to": str(resolved_to),
            "counts": {
                "games_scanned": len(games),
                "games_predicted": games_predicted,
                "games_skipped": games_skipped,
                "rows_upserted": rows_upserted,
                "paired_rows": len(paired_rows),
            },
            "skip_reasons": skip_reasons,
            "proof_summary": proof_summary,
            "proved_better": bool(proof_summary.get("proved_better")),
        }

        report_path = _write_report_file(run_id=run_id, report_payload=report_payload)
        report_payload["report_path"] = report_path

        run_row.status = status
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.games_scanned = len(games)
        run_row.games_predicted = games_predicted
        run_row.games_skipped = games_skipped
        run_row.rows_upserted = rows_upserted
        run_row.proof_json = report_payload
        db.commit()
        db.refresh(run_row)

        return {
            "run_id": str(run_id),
            "status": run_row.status,
            "active_model_version": run_row.active_model_version,
            "games_scanned": run_row.games_scanned,
            "games_predicted": run_row.games_predicted,
            "games_skipped": run_row.games_skipped,
            "rows_upserted": run_row.rows_upserted,
            "skip_reasons": skip_reasons,
            "proof_summary": proof_summary,
        }
    except Exception as exc:  # noqa: BLE001
        run_row.status = "failed"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.error_text = str(exc)
        db.commit()
        raise


def get_replay_report(db: Session, run_id: str) -> dict[str, Any]:
    try:
        parsed = uuid.UUID(run_id)
    except ValueError as exc:
        raise ReplayValidationError("run_id must be a valid UUID.") from exc

    run = db.get(WhlReplayRun, parsed)
    if run is None:
        raise ReplayNotFoundError(f"Replay run not found: {run_id}")

    proof = run.proof_json or {}
    skip_reasons = proof.get("skip_reasons") if isinstance(proof, dict) else {}
    proof_summary = proof.get("proof_summary") if isinstance(proof, dict) else None

    return {
        "run_id": str(run.id),
        "status": run.status,
        "date_from": run.date_from,
        "date_to": run.date_to,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "active_model_version": run.active_model_version,
        "games_scanned": run.games_scanned,
        "games_predicted": run.games_predicted,
        "games_skipped": run.games_skipped,
        "rows_upserted": run.rows_upserted,
        "skip_reasons": skip_reasons or {},
        "proof_summary": proof_summary,
        "error_text": run.error_text,
    }
