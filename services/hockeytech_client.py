from __future__ import annotations

import datetime as dt
import time
from typing import Any

import httpx

from core.config import settings


class HockeyTechClientError(RuntimeError):
    pass


class HockeyTechAccessDeniedError(HockeyTechClientError):
    pass


class HockeyTechClient:
    def __init__(
        self,
        client_code: str = "whl",
        api_key: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        timeout = float(timeout_seconds or settings.chl_ingest_timeout_seconds or 30)
        self._client = httpx.Client(base_url=settings.hockeytech_base_url, timeout=timeout)
        self._client_code = (client_code or "whl").strip().lower()
        self._api_key = (api_key or "").strip()

    @staticmethod
    def _response_preview(response: httpx.Response, max_len: int = 200) -> str:
        return (response.text or "").strip().replace("\n", " ")[:max_len]

    def _resolve_api_key(self) -> str:
        key = self._api_key or (settings.hockeytech_api_key or "").strip()
        if not key:
            raise HockeyTechClientError("HockeyTech API key is missing.")
        return key

    def _request(self, params: dict[str, Any], client_code: str | None = None) -> dict[str, Any]:
        resolved_client_code = (client_code or self._client_code)
        payload = {
            "client_code": resolved_client_code,
            "fmt": "json",
            "lang_code": "en",
            "key": self._resolve_api_key(),
            **params,
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.get("", params=payload)
                response.raise_for_status()

                content_type = (response.headers.get("content-type") or "").lower()
                if "application/json" not in content_type:
                    preview = self._response_preview(response)
                    if "client access denied" in preview.lower():
                        raise HockeyTechAccessDeniedError(
                            f"Client access denied for client_code={resolved_client_code}. "
                            "Check HockeyTech key entitlement."
                        )
                    raise HockeyTechClientError(
                        f"Unexpected HockeyTech content-type={content_type} for client_code={resolved_client_code}. "
                        f"Body preview: {preview}"
                    )

                try:
                    return response.json()
                except ValueError as exc:
                    preview = self._response_preview(response)
                    if "client access denied" in preview.lower():
                        raise HockeyTechAccessDeniedError(
                            f"Client access denied for client_code={resolved_client_code}. "
                            "Check HockeyTech key entitlement."
                        ) from exc
                    raise HockeyTechClientError(
                        f"Invalid JSON from HockeyTech for client_code={resolved_client_code}. "
                        f"Body preview: {preview}"
                    ) from exc
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, HockeyTechAccessDeniedError):
                    raise
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
        client_code: str | None = None,
    ) -> list[dict[str, Any]]:
        data = self._request(
            {
                "feed": "modulekit",
                "view": "scorebar",
                "numberofdaysahead": number_of_days_ahead,
                "numberofdaysback": number_of_days_back,
                "season_id": "",
                "team_id": str(team_id or ""),
            },
            client_code=client_code,
        )
        return data.get("SiteKit", {}).get("Scorebar", []) or []

    def get_seasons(self, client_code: str | None = None) -> list[dict[str, Any]]:
        data = self._request(
            {
                "feed": "modulekit",
                "view": "seasons",
            },
            client_code=client_code,
        )
        return data.get("SiteKit", {}).get("Seasons", []) or []

    def get_teams_by_season(self, season_id: str | int, client_code: str | None = None) -> list[dict[str, Any]]:
        data = self._request(
            {
                "feed": "modulekit",
                "view": "teamsbyseason",
                "season_id": str(season_id),
            },
            client_code=client_code,
        )
        return data.get("SiteKit", {}).get("Teamsbyseason", []) or []

    def get_schedule_by_season(self, season_id: str | int, client_code: str | None = None) -> list[dict[str, Any]]:
        data = self._request(
            {
                "feed": "modulekit",
                "view": "schedule",
                "season_id": str(season_id),
            },
            client_code=client_code,
        )
        return data.get("SiteKit", {}).get("Schedule", []) or []

    def get_clock(self, game_id: int, client_code: str | None = None) -> dict[str, Any]:
        data = self._request(
            {
                "feed": "gc",
                "game_id": game_id,
                "tab": "clock",
            },
            client_code=client_code,
        )
        return data.get("GC", {}).get("Clock", {}) or {}

    def get_gamesummary(self, game_id: int, client_code: str | None = None) -> dict[str, Any]:
        data = self._request(
            {
                "feed": "gc",
                "game_id": game_id,
                "tab": "gamesummary",
            },
            client_code=client_code,
        )
        return data.get("GC", {}).get("Gamesummary", {}) or {}

    def get_schedule_for_date(self, target_date: dt.date, client_code: str | None = None) -> list[dict[str, Any]]:
        # Pull a small window around tomorrow to avoid missing timezone edges.
        games = self.get_scorebar(number_of_days_ahead=2, number_of_days_back=0, client_code=client_code)
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
