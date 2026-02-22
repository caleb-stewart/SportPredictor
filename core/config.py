from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SportPredictor FastAPI"
    api_prefix: str = ""
    log_level: str = "INFO"
    api_port: int = 3141

    database_url: str = Field(
        default="postgresql+psycopg2://postgres:qqqq@localhost:5432/sportpredictor_development"
    )

    hockeytech_api_key: str = Field(default="", alias="HOCKEYTECH_API")
    hockeytech_api_key_whl: str = Field(default="", alias="HOCKEYTECH_API_KEY_WHL")
    hockeytech_api_key_ohl: str = Field(default="", alias="HOCKEYTECH_API_KEY_OHL")
    hockeytech_api_key_lhjmq: str = Field(default="", alias="HOCKEYTECH_API_KEY_LHJMQ")
    hockeytech_base_url: str = "https://lscluster.hockeytech.com/feed/"
    chl_ingest_max_workers: int = 6
    chl_ingest_rps_limit: float = 4.0
    chl_ingest_retry_max: int = 5
    chl_ingest_timeout_seconds: int = 30

    scheduler_enabled: bool = True
    scheduler_timezone: str = "America/Los_Angeles"

    model_store_root: str = str(Path(__file__).resolve().parents[1] / "model_store" / "whl_v2")
    active_model_file: str = "active_model.json"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    data_backend: str = "chl"
    dual_write: bool = False
    default_league_code: str = "whl"

    min_accuracy_gate: float = 0.60
    min_season_accuracy_gate: float = 0.55
    min_season_games_gate: int = 100
    weekly_compare_enabled: bool = True
    weekly_compare_baseline_model_version: str = "20260217T025309Z"
    weekly_compare_lookback_days: int = 365

    @cached_property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @cached_property
    def data_backend_normalized(self) -> str:
        normalized = (self.data_backend or "chl").strip().lower()
        return "chl" if normalized != "chl" else normalized


settings = Settings()
