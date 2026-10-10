from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any

import httpx

from core.config import settings


class NhlClientError(RuntimeError):
    """Raised when the public NHL API or an NHL HTML report cannot be read."""


class _RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / max(float(requests_per_second), 0.1)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._interval - (now - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()


def nhl_report_url(game_id: int | str, report_code: str, *, base_url: str | None = None) -> str:
    """Build the official NHL GS/FS report URL for a GameCenter id."""
    game_id_text = str(game_id).strip()
    if len(game_id_text) < 10 or not game_id_text.isdigit():
        raise NhlClientError(f"Unexpected NHL game id: {game_id!r}")
    season_start = int(game_id_text[:4])
    season_directory = f"{season_start}{season_start + 1}"
    report_code = report_code.strip().upper()
    if report_code not in {"GS", "FS"}:
        raise NhlClientError(f"Unsupported NHL report code: {report_code!r}")
    root = (base_url or settings.nhl_report_base_url).rstrip("/")
    return f"{root}/{season_directory}/{report_code}{game_id_text[-6:]}.HTM"


class NhlClient:
    """Small client for the official, unauthenticated NHL GameCenter API."""

    def __init__(
        self,
        *,
        timeout_seconds: int | None = None,
        requests_per_second: float | None = None,
    ) -> None:
        timeout = float(timeout_seconds or settings.nhl_ingest_timeout_seconds or 30)
        self._client = httpx.Client(base_url=settings.nhl_api_base_url.rstrip("/"), timeout=timeout)
        self._limiter = _RateLimiter(requests_per_second or settings.nhl_ingest_rps_limit or 3.0)
        self._schedule_cache: dict[str, list[dict[str, Any]]] = {}

    def close(self) -> None:
        """Release the HTTP connection pool after a batch import."""
        self._client.close()

    def __enter__(self) -> "NhlClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _preview(response: httpx.Response, max_len: int = 240) -> str:
        return (response.text or "").strip().replace("\n", " ")[:max_len]

    def _request_json(self, path: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self._limiter.wait()
                response = self._client.get(path)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise NhlClientError(f"Expected an object from NHL API at {path}")
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise NhlClientError(f"NHL API request failed after retries path={path}: {last_error}") from last_error

    def _request_report(self, game_id: int | str, report_code: str) -> str:
        url = nhl_report_url(game_id, report_code)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self._limiter.wait()
                response = self._client.get(url)
                # Missing historical reports are expected in the NHL archive;
                # retrying a permanent 404 only slows a large backfill.
                if response.status_code == 404:
                    raise NhlClientError(f"NHL report not found url={url}")
                response.raise_for_status()
                return response.text
            except NhlClientError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise NhlClientError(f"NHL report request failed after retries url={url}: {last_error}") from last_error

    @staticmethod
    def _extract_games(payload: Any) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("id") is not None and isinstance(value.get("homeTeam"), dict) and isinstance(value.get("awayTeam"), dict):
                    found[str(value["id"])] = value
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        return list(found.values())

    def get_schedule_for_date(self, target_date: dt.date) -> list[dict[str, Any]]:
        target_date_text = target_date.isoformat()
        if target_date_text in self._schedule_cache:
            return self._schedule_cache[target_date_text]

        payload = self._request_json(f"/v1/schedule/{target_date.isoformat()}")
        # The schedule endpoint returns a seven-day gameWeek whose nested game
        # objects omit gameDate. Normalize and cache the whole week so a long
        # historical import does not issue one schedule request per day.
        game_week = payload.get("gameWeek") if isinstance(payload, dict) else None
        if isinstance(game_week, list):
            for day in game_week:
                if not isinstance(day, dict) or not day.get("date"):
                    continue
                day_date = str(day["date"])
                day_games: list[dict[str, Any]] = []
                for game in day.get("games") or []:
                    if not isinstance(game, dict):
                        continue
                    normalized = dict(game)
                    normalized.setdefault("gameDate", day_date)
                    day_games.append(normalized)
                self._schedule_cache[day_date] = day_games
            if target_date_text in self._schedule_cache:
                return self._schedule_cache[target_date_text]

        games = [game for game in self._extract_games(payload) if game.get("gameDate") == target_date_text]
        if games:
            self._schedule_cache[target_date_text] = games
            return games

        # The score endpoint is a useful fallback when the schedule feed has
        # changed shape or is temporarily missing a date's games.
        score_payload = self._request_json(f"/v1/score/{target_date.isoformat()}")
        # Never accept a missing date here: the score feed can include games
        # from another season without a gameDate field.
        games = [game for game in self._extract_games(score_payload) if game.get("gameDate") == target_date_text]
        self._schedule_cache[target_date_text] = games
        return games

    def get_landing(self, game_id: int | str) -> dict[str, Any]:
        return self._request_json(f"/v1/gamecenter/{game_id}/landing")

    def get_boxscore(self, game_id: int | str) -> dict[str, Any]:
        return self._request_json(f"/v1/gamecenter/{game_id}/boxscore")

    def get_play_by_play(self, game_id: int | str) -> dict[str, Any]:
        return self._request_json(f"/v1/gamecenter/{game_id}/play-by-play")

    def get_game_summary_report(self, game_id: int | str) -> str:
        return self._request_report(game_id, "GS")

    def get_faceoff_summary_report(self, game_id: int | str) -> str:
        return self._request_report(game_id, "FS")
