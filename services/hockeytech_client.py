from __future__ import annotations

import datetime as dt
import time
from typing import Any, Optional

import httpx

from core.config import settings


class HockeyTechClientError(RuntimeError):
    pass


class HockeyTechClient:
    def __init__(self) -> None:
        self._client = httpx.Client(base_url=settings.hockeytech_base_url, timeout=30)

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        if not settings.hockeytech_api_key:
            raise HockeyTechClientError("HOCKEYTECH_API key is missing.")

        payload = {
            "client_code": "whl",
            "fmt": "json",
            "lang_code": "en",
            "key": settings.hockeytech_api_key,
            **params,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = self._client.get("", params=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue

        raise HockeyTechClientError(f"HockeyTech request failed after retries: {last_exc}")

    def get_scorebar(
        self,
        number_of_days_ahead: int = 0,
        number_of_days_back: int = 0,
        team_id: str | int | None = None,
    ) -> list[dict[str, Any]]:
        data = self._request(
            {
                "feed": "modulekit",
                "view": "scorebar",
                "numberofdaysahead": number_of_days_ahead,
                "numberofdaysback": number_of_days_back,
                "season_id": "",
                "team_id": str(team_id or ""),
            }
        )
        return data.get("SiteKit", {}).get("Scorebar", []) or []

    def get_clock(self, game_id: int) -> dict[str, Any]:
        data = self._request(
            {
                "feed": "gc",
                "game_id": game_id,
                "tab": "clock",
            }
        )
        return data.get("GC", {}).get("Clock", {}) or {}

    def get_schedule_for_date(self, target_date: dt.date) -> list[dict[str, Any]]:
        # Pull a small window around tomorrow to avoid missing timezone edges.
        games = self.get_scorebar(number_of_days_ahead=2, number_of_days_back=0)
        filtered: list[dict[str, Any]] = []

        for game in games:
            iso_str = game.get("GameDateISO8601")
            if not iso_str:
                continue
            try:
                game_dt = dt.datetime.fromisoformat(iso_str)
            except ValueError:
                continue
            if game_dt.date() == target_date:
                filtered.append(game)

        return filtered
