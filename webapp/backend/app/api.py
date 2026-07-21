from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .services.data_service import (
    basis_payload,
    comparison_payload,
    drivers_payload,
    forecast_payload,
    ingestion_status,
    load_master,
    load_registry,
    price_history,
    reliability_payload,
    signal_payload,
)


router = APIRouter(prefix=settings.api_prefix)


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "database": settings.database_url.split(":", 1)[0],
        "data_available": settings.master_data_path.exists(),
        "registry_available": settings.model_registry_path.exists(),
    }


@router.get("/meta")
def meta() -> dict:
    frame = load_master()
    registry = load_registry()
    return {
        "project": "CofPred",
        "target": "Daily Vietnamese domestic Robusta farm-gate price",
        "raw_rows": len(frame),
        "series_rows": int(frame["Gia_target"].notna().sum()),
        "date_min": frame["Ngay"].min().date().isoformat(),
        "date_max": frame["Ngay"].max().date().isoformat(),
        "deployed_models": sorted(registry["model"].dropna().unique().tolist()) if not registry.empty else [],
        "deployed_horizons": sorted(registry["horizon"].dropna().astype(int).unique().tolist()) if not registry.empty else [],
        "research_models": 22,
        "research_horizons": [1, 5, 21, 63],
    }


@router.get("/prices")
def prices(days: int = Query(365, ge=1, le=2000)) -> dict:
    return {"unit": "VND/kg", "data": price_history(days)}


@router.get("/forecast")
def forecast(
    model: str = Query("best"),
    horizon: int = Query(5),
    history_days: int = Query(180, ge=30, le=1000),
) -> dict:
    try:
        return forecast_payload(model, horizon, history_days)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/signal")
def signal(threshold_pct: float = Query(1.0, ge=0.0, le=20.0)) -> dict:
    return signal_payload(threshold_pct)


@router.get("/model-comparison")
def model_comparison(horizon: int = Query(5)) -> dict:
    try:
        return comparison_payload(horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/basis")
def basis(days: int = Query(365, ge=30, le=2000)) -> dict:
    return basis_payload(days)


@router.get("/drivers")
def drivers(days: int = Query(365, ge=30, le=2000)) -> dict:
    return drivers_payload(days)


@router.get("/reliability")
def reliability(horizon: int = Query(5)) -> dict:
    try:
        return reliability_payload(horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/mcs")
def mcs(horizon: int = Query(5)) -> dict:
    result = reliability(horizon)
    return {"horizon": horizon, **result["mcs"]}


@router.get("/ingestion-status")
def get_ingestion_status(db: Session = Depends(get_db)) -> dict:
    return ingestion_status(db)

