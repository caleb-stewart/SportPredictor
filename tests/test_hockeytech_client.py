from __future__ import annotations

import pytest

from services.hockeytech_client import HockeyTechAccessDeniedError, HockeyTechClient, HockeyTechClientError


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, content_type: str = "application/json; charset=utf-8", text: str = "{}", json_data=None):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = text
        self._json_data = json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._json_data is None:
            raise ValueError("invalid json")
        return self._json_data


def test_access_denied_non_json_raises_typed_error():
    client = HockeyTechClient(client_code="ohl", api_key="abc")
    denied = _FakeResponse(content_type="text/html; charset=UTF-8", text="Client access denied.")
    client._client.get = lambda *_args, **_kwargs: denied  # type: ignore[method-assign]

    with pytest.raises(HockeyTechAccessDeniedError):
        client._request({"feed": "modulekit", "view": "seasons"})


def test_unexpected_non_json_raises_client_error():
    client = HockeyTechClient(client_code="whl", api_key="abc")
    bad = _FakeResponse(content_type="text/html; charset=UTF-8", text="<html>oops</html>")
    client._client.get = lambda *_args, **_kwargs: bad  # type: ignore[method-assign]

    with pytest.raises(HockeyTechClientError):
        client._request({"feed": "modulekit", "view": "seasons"})


def test_get_seasons_parses_sitekit_payload():
    client = HockeyTechClient(client_code="whl", api_key="abc")
    client._request = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "SiteKit": {
            "Seasons": [
                {"season_id": "289", "season_name": "2025 - 26 Regular Season"},
            ]
        }
    }

    seasons = client.get_seasons()
    assert len(seasons) == 1
    assert seasons[0]["season_id"] == "289"
