from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class PriceObservation(Base):
    __tablename__ = "price_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observed_on: Mapped[date] = mapped_column(Date, unique=True, index=True)
    domestic_price: Mapped[float] = mapped_column(Float)
    london_vnd_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    usd_vnd: Mapped[float | None] = mapped_column(Float, nullable=True)
    diesel: Mapped[float | None] = mapped_column(Float, nullable=True)
    rain_90d: Mapped[float | None] = mapped_column(Float, nullable=True)
    oni: Mapped[float | None] = mapped_column(Float, nullable=True)
    water_balance_90d: Mapped[float | None] = mapped_column(Float, nullable=True)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(20))
    rows_loaded: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[str | None] = mapped_column(String(500), nullable=True)
    finished_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

