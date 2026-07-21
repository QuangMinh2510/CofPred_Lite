from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class PricePoint(BaseModel):
    date: date
    value: float

    model_config = ConfigDict(from_attributes=True)


class ForecastPoint(BaseModel):
    date: date
    forecast: float
    lower_50: float
    upper_50: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class HealthResponse(BaseModel):
    status: str
    database: str
    data_available: bool
    registry_available: bool

