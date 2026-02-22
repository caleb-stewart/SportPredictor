from __future__ import annotations

import pytest

from services.data_backend import hockeytech_api_key_for_league


def test_hockeytech_api_key_prefers_per_league(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "hockeytech_api_key_whl", "whl-key")
    monkeypatch.setattr(settings, "hockeytech_api_key", "fallback-key")

    assert hockeytech_api_key_for_league("whl") == "whl-key"


def test_hockeytech_api_key_falls_back_to_global(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "hockeytech_api_key_ohl", "")
    monkeypatch.setattr(settings, "hockeytech_api_key", "fallback-key")

    assert hockeytech_api_key_for_league("ohl") == "fallback-key"


def test_hockeytech_api_key_raises_if_all_missing(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "hockeytech_api_key_whl", "")
    monkeypatch.setattr(settings, "hockeytech_api_key_ohl", "")
    monkeypatch.setattr(settings, "hockeytech_api_key_lhjmq", "")
    monkeypatch.setattr(settings, "hockeytech_api_key", "")

    with pytest.raises(RuntimeError):
        hockeytech_api_key_for_league("whl")
