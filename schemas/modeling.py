from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    active: bool
    model_version: Optional[str] = None
    model_family: Optional[str] = None
    k_values: Optional[list[int]] = None
    error: Optional[str] = None


class ModelTrainResponse(BaseModel):
    ok: bool
    promoted: bool
    details: dict[str, Any]
