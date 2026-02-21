from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.games import router as games_router
from api.routes.models import router as models_router
from api.routes.predictions import router as predictions_router
from api.routes.teams import router as teams_router
from core.config import settings
from jobs.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(games_router)
app.include_router(predictions_router)
app.include_router(models_router)
app.include_router(teams_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    if settings.scheduler_enabled:
        start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_scheduler()
