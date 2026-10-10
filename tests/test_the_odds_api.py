from __future__ import annotations

import datetime as dt
import json

import httpx

from services.the_odds_api import TheOddsApiClient, _is_live_at


def _client(handler) -> TheOddsApiClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://odds.example", transport=transport)
    return TheOddsApiClient(api_key="test-key", client=http_client)


def test_current_odds_requests_normalized_pregame_markets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v4/sports/icehockey_nhl/odds"
        assert request.url.params["regions"] == "us"
        assert request.url.params["markets"] == "h2h,spreads,totals"
        assert request.url.params["oddsFormat"] == "decimal"
        return httpx.Response(
            200,
            headers={"x-requests-remaining": "497", "x-requests-used": "3", "x-requests-last": "3"},
            json=[{"id": "event-1", "sport_key": "icehockey_nhl"}],
        )

    client = _client(handler)
    response = client.current_odds("icehockey_nhl")

    assert response.data[0]["id"] == "event-1"
    assert response.quota_remaining == 497
    assert response.quota_cost == 3


def test_historical_odds_uses_requested_snapshot_and_parses_response_times() -> None:
    requested_at = dt.datetime(2025, 10, 1, tzinfo=dt.UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v4/historical/sports/icehockey_nhl/odds"
        assert request.url.params["date"] == "2025-10-01T00:00:00Z"
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "timestamp": "2025-10-01T00:00:00Z",
                    "next_timestamp": "2025-10-01T00:05:00Z",
                    "data": [],
                }
            ),
        )

    client = _client(handler)
    response = client.historical_odds("icehockey_nhl", requested_at)

    assert response.snapshot_at == requested_at
    assert response.next_snapshot_at == requested_at + dt.timedelta(minutes=5)


def test_historical_quote_liveness_uses_source_snapshot_not_import_time() -> None:
    commence = dt.datetime(2025, 10, 1, 19, tzinfo=dt.UTC)
    assert _is_live_at(commence, dt.datetime(2025, 10, 1, 18, tzinfo=dt.UTC)) is False
    assert _is_live_at(commence, dt.datetime(2025, 10, 1, 19, tzinfo=dt.UTC)) is True
