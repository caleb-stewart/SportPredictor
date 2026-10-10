"""Durable ingestion for The Odds API's pregame sportsbook feed.

This module intentionally never places orders or requires a betting account. It
records every returned sportsbook outcome as an append-only quote so later
research can reproduce the market that was available to the model.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import settings
from db.models import (
    MarketDataEvent,
    MarketDataIngestionRun,
    MarketDataProvider,
    MarketDataQuote,
)

PROVIDER_CODE = "the_odds_api"
PROVIDER_NAME = "The Odds API"


class TheOddsApiError(RuntimeError):
    """Raised when upstream odds data cannot safely be retrieved or parsed."""


@dataclass(frozen=True)
class OddsApiResponse:
    data: list[dict[str, Any]]
    quota_remaining: int | None
    quota_used: int | None
    quota_cost: int | None
    snapshot_at: dt.datetime | None = None
    next_snapshot_at: dt.datetime | None = None


@dataclass(frozen=True)
class OddsSyncResult:
    run_id: uuid.UUID
    sport_key: str
    operation: str
    events_written: int
    quotes_written: int
    quota_remaining: int | None
    quota_cost: int | None
    snapshot_at: dt.datetime | None = None
    next_snapshot_at: dt.datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "provider": PROVIDER_CODE,
            "sport_key": self.sport_key,
            "operation": self.operation,
            "events_written": self.events_written,
            "quotes_written": self.quotes_written,
            "quota_remaining": self.quota_remaining,
            "quota_cost": self.quota_cost,
            "snapshot_at": self.snapshot_at,
            "next_snapshot_at": self.next_snapshot_at,
        }


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        try:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise TheOddsApiError(f"Invalid timestamp from The Odds API: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _parse_header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class TheOddsApiClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else settings.the_odds_api_key).strip()
        if not self._api_key:
            raise TheOddsApiError("THE_ODDS_API_KEY is missing.")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=(base_url or settings.the_odds_api_base_url).rstrip("/"),
            timeout=float(timeout_seconds or settings.the_odds_api_timeout_seconds),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, *, params: dict[str, Any]) -> tuple[Any, httpx.Headers]:
        try:
            response = self._client.get(path, params={"apiKey": self._api_key, **params})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            preview = (exc.response.text or "").replace("\n", " ").strip()[:300]
            raise TheOddsApiError(
                f"The Odds API returned HTTP {exc.response.status_code}: {preview}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TheOddsApiError(f"The Odds API request failed: {exc}") from exc

        try:
            return response.json(), response.headers
        except ValueError as exc:
            raise TheOddsApiError("The Odds API returned invalid JSON.") from exc

    @staticmethod
    def _quota(headers: httpx.Headers) -> tuple[int | None, int | None, int | None]:
        return (
            _parse_header_int(headers, "x-requests-remaining"),
            _parse_header_int(headers, "x-requests-used"),
            _parse_header_int(headers, "x-requests-last"),
        )

    def list_sports(self) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
        payload, headers = self._get("/v4/sports", params={})
        if not isinstance(payload, list):
            raise TheOddsApiError("The Odds API sports response was not a list.")
        remaining, used, cost = self._quota(headers)
        return payload, {"quota_remaining": remaining, "quota_used": used, "quota_cost": cost}

    def current_odds(
        self,
        sport_key: str,
        *,
        regions: str | None = None,
        markets: str | None = None,
    ) -> OddsApiResponse:
        payload, headers = self._get(
            f"/v4/sports/{sport_key}/odds",
            params={
                "regions": regions or settings.the_odds_api_regions,
                "markets": markets or settings.the_odds_api_markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if not isinstance(payload, list):
            raise TheOddsApiError("The Odds API current-odds response was not a list.")
        remaining, used, cost = self._quota(headers)
        return OddsApiResponse(payload, remaining, used, cost)

    def historical_odds(
        self,
        sport_key: str,
        snapshot_at: dt.datetime,
        *,
        regions: str | None = None,
        markets: str | None = None,
    ) -> OddsApiResponse:
        snapshot_at = snapshot_at.astimezone(dt.UTC)
        payload, headers = self._get(
            f"/v4/historical/sports/{sport_key}/odds",
            params={
                "date": snapshot_at.isoformat().replace("+00:00", "Z"),
                "regions": regions or settings.the_odds_api_regions,
                "markets": markets or settings.the_odds_api_markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise TheOddsApiError("The Odds API historical-odds response had no data list.")
        remaining, used, cost = self._quota(headers)
        return OddsApiResponse(
            data=payload["data"],
            quota_remaining=remaining,
            quota_used=used,
            quota_cost=cost,
            snapshot_at=_parse_timestamp(payload.get("timestamp")),
            next_snapshot_at=_parse_timestamp(payload.get("next_timestamp")),
        )


def _provider(db: Session) -> MarketDataProvider:
    row = db.scalar(select(MarketDataProvider).where(MarketDataProvider.code == PROVIDER_CODE))
    if row is None:
        row = MarketDataProvider(code=PROVIDER_CODE, name=PROVIDER_NAME)
        db.add(row)
        db.flush()
    return row


def _event(db: Session, provider: MarketDataProvider, payload: dict[str, Any], observed_at: dt.datetime) -> MarketDataEvent:
    provider_event_id = str(payload.get("id") or "").strip()
    if not provider_event_id:
        raise TheOddsApiError("The Odds API event did not contain an id.")
    row = db.scalar(
        select(MarketDataEvent).where(
            MarketDataEvent.provider_id == provider.id,
            MarketDataEvent.provider_event_id == provider_event_id,
        )
    )
    fields = {
        "sport_key": str(payload.get("sport_key") or ""),
        "sport_title": payload.get("sport_title"),
        "commence_time": _parse_timestamp(payload.get("commence_time")),
        "home_team": payload.get("home_team"),
        "away_team": payload.get("away_team"),
        "last_seen_at": observed_at,
        "latest_payload": payload,
    }
    if not fields["sport_key"]:
        raise TheOddsApiError(f"The Odds API event {provider_event_id} did not contain sport_key.")
    if row is None:
        row = MarketDataEvent(
            provider_id=provider.id,
            provider_event_id=provider_event_id,
            first_seen_at=observed_at,
            **fields,
        )
        db.add(row)
        db.flush()
    else:
        for name, value in fields.items():
            setattr(row, name, value)
    return row


def _decimal(value: Any, *, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TheOddsApiError(f"Invalid {field_name} value from The Odds API: {value!r}") from exc


def _is_live_at(commence_time: dt.datetime | None, captured_at: dt.datetime) -> bool:
    """Classify the market at its source snapshot, never at import time."""
    return bool(commence_time and commence_time <= captured_at)


def _store_response(
    db: Session,
    response: OddsApiResponse,
    *,
    captured_at: dt.datetime,
) -> tuple[int, int]:
    provider = _provider(db)
    events_written = 0
    quotes_written = 0
    for event_payload in response.data:
        if not isinstance(event_payload, dict):
            raise TheOddsApiError("The Odds API returned a non-object event.")
        event = _event(db, provider, event_payload, observed_at=captured_at)
        events_written += 1
        is_live = _is_live_at(event.commence_time, captured_at)
        for bookmaker in event_payload.get("bookmakers") or []:
            if not isinstance(bookmaker, dict):
                continue
            bookmaker_key = str(bookmaker.get("key") or "").strip()
            if not bookmaker_key:
                continue
            source_updated_at = _parse_timestamp(bookmaker.get("last_update"))
            for market in bookmaker.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                market_key = str(market.get("key") or "").strip()
                if not market_key:
                    continue
                for outcome in market.get("outcomes") or []:
                    if not isinstance(outcome, dict) or outcome.get("price") is None:
                        continue
                    quote = MarketDataQuote(
                        event_id=event.id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker.get("title"),
                        market_key=market_key,
                        outcome_name=str(outcome.get("name") or "").strip(),
                        outcome_description=outcome.get("description"),
                        point=_decimal(outcome["point"], field_name="point") if outcome.get("point") is not None else None,
                        decimal_price=_decimal(outcome["price"], field_name="price"),
                        source_updated_at=source_updated_at,
                        captured_at=captured_at,
                        is_live=is_live,
                        raw_quote={"bookmaker": bookmaker, "market": market, "outcome": outcome},
                    )
                    if quote.outcome_name:
                        db.add(quote)
                        quotes_written += 1
    return events_written, quotes_written


def _sync(
    db: Session,
    *,
    sport_key: str,
    operation: str,
    response: OddsApiResponse,
    captured_at: dt.datetime,
) -> OddsSyncResult:
    run = MarketDataIngestionRun(
        provider_code=PROVIDER_CODE,
        operation=operation,
        sport_key=sport_key,
        status="running",
    )
    db.add(run)
    db.commit()
    try:
        events_written, quotes_written = _store_response(db, response, captured_at=captured_at)
        run.status = "completed"
        run.completed_at = dt.datetime.now(dt.UTC)
        run.events_written = events_written
        run.quotes_written = quotes_written
        run.quota_remaining = response.quota_remaining
        run.quota_used = response.quota_used
        run.quota_cost = response.quota_cost
        run.details = {
            "snapshot_at": response.snapshot_at.isoformat() if response.snapshot_at else None,
            "next_snapshot_at": response.next_snapshot_at.isoformat() if response.next_snapshot_at else None,
        }
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        failed_run = db.get(MarketDataIngestionRun, run.id)
        if failed_run is not None:
            failed_run.status = "failed"
            failed_run.completed_at = dt.datetime.now(dt.UTC)
            failed_run.error_text = str(exc)[:1000]
            db.commit()
        if isinstance(exc, TheOddsApiError):
            raise
        raise TheOddsApiError(f"Failed to persist The Odds API data: {exc}") from exc
    return OddsSyncResult(
        run_id=run.id,
        sport_key=sport_key,
        operation=operation,
        events_written=events_written,
        quotes_written=quotes_written,
        quota_remaining=response.quota_remaining,
        quota_cost=response.quota_cost,
        snapshot_at=response.snapshot_at,
        next_snapshot_at=response.next_snapshot_at,
    )


def sync_current_odds(
    db: Session,
    sport_key: str,
    *,
    client: TheOddsApiClient | None = None,
) -> OddsSyncResult:
    owned_client = client is None
    client = client or TheOddsApiClient()
    try:
        response = client.current_odds(sport_key)
        return _sync(
            db,
            sport_key=sport_key,
            operation="current_odds",
            response=response,
            captured_at=dt.datetime.now(dt.UTC),
        )
    finally:
        if owned_client:
            client.close()


def sync_historical_odds_snapshot(
    db: Session,
    sport_key: str,
    snapshot_at: dt.datetime,
    *,
    client: TheOddsApiClient | None = None,
) -> OddsSyncResult:
    owned_client = client is None
    client = client or TheOddsApiClient()
    try:
        response = client.historical_odds(sport_key, snapshot_at)
        return _sync(
            db,
            sport_key=sport_key,
            operation="historical_odds",
            response=response,
            captured_at=response.snapshot_at or snapshot_at.astimezone(dt.UTC),
        )
    finally:
        if owned_client:
            client.close()


def market_data_status(db: Session) -> dict[str, Any]:
    latest_run = db.scalar(
        select(MarketDataIngestionRun)
        .where(MarketDataIngestionRun.provider_code == PROVIDER_CODE)
        .order_by(MarketDataIngestionRun.started_at.desc())
    )
    return {
        "provider": PROVIDER_CODE,
        "configured": bool(settings.the_odds_api_key),
        "current_sports": [item.strip() for item in settings.the_odds_api_current_sports.split(",") if item.strip()],
        "markets": [item.strip() for item in settings.the_odds_api_markets.split(",") if item.strip()],
        "regions": [item.strip() for item in settings.the_odds_api_regions.split(",") if item.strip()],
        "event_count": int(db.scalar(select(func.count()).select_from(MarketDataEvent)) or 0),
        "quote_count": int(db.scalar(select(func.count()).select_from(MarketDataQuote)) or 0),
        "latest_run": {
            "id": str(latest_run.id),
            "operation": latest_run.operation,
            "sport_key": latest_run.sport_key,
            "status": latest_run.status,
            "started_at": latest_run.started_at,
            "completed_at": latest_run.completed_at,
            "events_written": latest_run.events_written,
            "quotes_written": latest_run.quotes_written,
            "quota_remaining": latest_run.quota_remaining,
            "quota_cost": latest_run.quota_cost,
            "error_text": latest_run.error_text,
        }
        if latest_run
        else None,
    }
