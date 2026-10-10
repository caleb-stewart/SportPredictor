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
    upcoming_prediction_refresh_minutes: int = Field(default=30, alias="UPCOMING_PREDICTION_REFRESH_MINUTES")
    local_maintenance_refresh_enabled: bool = Field(default=False, alias="LOCAL_MAINTENANCE_REFRESH_ENABLED")
    local_maintenance_max_catchup_days: int = Field(default=14, alias="LOCAL_MAINTENANCE_MAX_CATCHUP_DAYS")

    model_store_root: str = str(Path(__file__).resolve().parents[1] / "model_store" / "whl_v2")
    active_model_file: str = "active_model.json"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    cors_origin_regex: str | None = Field(
        default=(
            r"^https?://("
            r"localhost|127\.0\.0\.1|"
            r"[a-zA-Z0-9-]+(?:\.local)?|"
            r"10(?:\.\d{1,3}){3}|"
            r"192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
            r")(?::\d+)?$"
        ),
        alias="CORS_ORIGIN_REGEX",
    )
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
    def cors_origin_regex_value(self) -> str | None:
        value = (self.cors_origin_regex or "").strip()
        if not value:
            return None
        # Allow either "\." or "\\." style escapes from .env files.
        return value.replace("\\\\", "\\")

    @cached_property
    def data_backend_normalized(self) -> str:
        normalized = (self.data_backend or "chl").strip().lower()
        return "chl" if normalized != "chl" else normalized


settings = Settings()
