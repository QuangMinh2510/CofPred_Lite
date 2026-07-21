from __future__ import annotations

from functools import lru_cache
from math import sqrt

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import IngestionRun, PriceObservation


TARGET = "Gia_target"
DRIVER_COLUMNS = {
    "USD/VND": "usdvnd_lag1",
    "Diesel": "diesel",
    "ENSO (ONI)": "oni",
    "Rain 90d": "rain_90d",
    "Water balance 90d": "waterbal_90d",
}
RESEARCH_SAMPLE = {1: 122, 5: 121, 21: 118, 63: 109}


@lru_cache(maxsize=1)
def load_master() -> pd.DataFrame:
    path = settings.master_data_path
    if not path.exists():
        raise FileNotFoundError(f"Master data not found: {path}")
    frame = pd.read_csv(path)
    frame["Ngay"] = pd.to_datetime(frame["Ngay"], errors="coerce")
    numeric = [TARGET, "london_vnd_kg_lag1", *DRIVER_COLUMNS.values()]
    for column in dict.fromkeys(numeric):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["Ngay"])
        .sort_values("Ngay")
        .drop_duplicates("Ngay", keep="last")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def load_registry() -> pd.DataFrame:
    path = settings.model_registry_path
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in ["horizon", "MAE", "RMSE", "MAPE_pct", "DA_pct", "latest_prediction"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@lru_cache(maxsize=1)
def load_research_comparison() -> pd.DataFrame:
    path = settings.master_data_path.parents[2] / "webapp" / "backend" / "app" / "data" / "model_mse.csv"
    frame = pd.read_csv(path)
    for column in ["h1", "h5", "h21", "h63"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def bootstrap_database(db: Session) -> int:
    frame = load_master().dropna(subset=[TARGET])
    existing_dates = set(db.scalars(select(PriceObservation.observed_on)).all())
    rows: list[PriceObservation] = []
    for record in frame.to_dict("records"):
        observed_on = pd.Timestamp(record["Ngay"]).date()
        if observed_on in existing_dates:
            continue
        rows.append(
            PriceObservation(
                observed_on=observed_on,
                domestic_price=float(record[TARGET]),
                london_vnd_kg=_optional_float(record.get("london_vnd_kg_lag1")),
                usd_vnd=_optional_float(record.get("usdvnd_lag1")),
                diesel=_optional_float(record.get("diesel")),
                rain_90d=_optional_float(record.get("rain_90d")),
                oni=_optional_float(record.get("oni")),
                water_balance_90d=_optional_float(record.get("waterbal_90d")),
            )
        )
    if rows:
        db.add_all(rows)
    db.add(IngestionRun(source="master_csv", status="ok", rows_loaded=len(rows)))
    db.commit()
    return len(rows)


def _optional_float(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def price_history(days: int = 365) -> list[dict]:
    frame = load_master().dropna(subset=[TARGET]).tail(max(1, min(days, 2000)))
    return [
        {"date": row.Ngay.date().isoformat(), "value": round(float(row.Gia_target), 2)}
        for row in frame.itertuples()
    ]


def _select_registry_model(model: str | None, horizon: int) -> pd.Series | None:
    registry = load_registry()
    if registry.empty:
        return None
    available = registry[registry["horizon"] == horizon].copy()
    if available.empty:
        return None
    if model and model.lower() != "best":
        selected = available[available["model"].str.lower() == model.lower()]
        if not selected.empty:
            return selected.sort_values(["RMSE", "MAE"]).iloc[0]
    return available.sort_values(["RMSE", "MAE"]).iloc[0]


def forecast_payload(model: str | None, horizon: int, history_days: int = 180) -> dict:
    if horizon not in (1, 5, 21, 63):
        raise ValueError("horizon must be one of 1, 5, 21 or 63")
    master = load_master().dropna(subset=[TARGET])
    latest = master.iloc[-1]
    selected = _select_registry_model(model, horizon)
    anchor = float(latest[TARGET])
    today = pd.Timestamp.now().normalize()
    reference_date = max(today, pd.Timestamp(latest["Ngay"]).normalize())

    if selected is None:
        model_name = "RandomWalk"
        point = anchor
        research = load_research_comparison()
        naive = research[research["model"] == "RandomWalk"].iloc[0]
        rmse = sqrt(float(naive[f"h{horizon}"]) * 1_000_000)
        source = "random-walk fallback; no deployed model for this horizon"
    else:
        model_name = str(selected["model"])
        point = float(selected["latest_prediction"])
        rmse = float(selected["RMSE"])
        source = "models/model_registry.csv"

    dates = pd.bdate_range(reference_date + BDay(1), periods=horizon)
    points = []
    quantiles = {50: 0.67449, 80: 1.28155, 95: 1.95996}
    for step, forecast_date in enumerate(dates, start=1):
        progress = step / horizon
        central = anchor + (point - anchor) * progress
        sigma = rmse * sqrt(progress)
        row = {"date": forecast_date.date().isoformat(), "forecast": round(central, 2)}
        for level, z_value in quantiles.items():
            row[f"lower_{level}"] = round(max(0.0, central - z_value * sigma), 2)
            row[f"upper_{level}"] = round(central + z_value * sigma, 2)
        points.append(row)

    return {
        "model": model_name,
        "horizon": horizon,
        "data_as_of": pd.Timestamp(latest["Ngay"]).date().isoformat(),
        "forecast_anchor_date": reference_date.date().isoformat(),
        "anchor_price": round(anchor, 2),
        "point_forecast": round(point, 2),
        "rmse": round(rmse, 2),
        "source": source,
        "history": price_history(history_days),
        "forecast": points,
        "disclaimer": "Decision support only; not investment advice.",
    }


def signal_payload(threshold_pct: float = 1.0) -> dict:
    payload = forecast_payload("best", 5, 60)
    change_pct = (payload["point_forecast"] / payload["anchor_price"] - 1) * 100
    signal = "LONG" if change_pct >= threshold_pct else "FLAT"
    if payload["model"] == "RandomWalk":
        reason = "No h=5 model is deployed, so the statistically defensible RandomWalk fallback is FLAT."
    else:
        reason = f"{payload['model']} forecasts a {change_pct:+.2f}% five-session change."
    return {
        "signal": signal,
        "horizon": 5,
        "threshold_pct": threshold_pct,
        "change_pct": round(change_pct, 3),
        "model": payload["model"],
        "reason": reason,
        "disclaimer": payload["disclaimer"],
    }


def comparison_payload(horizon: int) -> dict:
    frame = load_research_comparison().copy()
    column = f"h{horizon}"
    if column not in frame.columns:
        raise ValueError("horizon must be one of 1, 5, 21 or 63")
    benchmark = float(frame.loc[frame["model"] == "RandomWalk", column].iloc[0])
    frame["mse"] = frame[column]
    frame["delta_vs_rw_pct"] = (frame["mse"] / benchmark - 1) * 100
    display = frame[["family", "model", "mse", "delta_vs_rw_pct"]].astype(object)
    display = display.where(pd.notna(display), None)
    records = display.to_dict("records")
    for row in records:
        if row["mse"] is not None:
            row["mse"] = round(float(row["mse"]), 2)
            row["delta_vs_rw_pct"] = round(float(row["delta_vs_rw_pct"]), 2)
    return {
        "horizon": horizon,
        "unit": "million (VND/kg)^2",
        "benchmark_mse": benchmark,
        "models": records,
        "interpretation": (
            "A lower point MSE does not by itself establish statistical superiority. "
            "The offline MCS analysis does not find robust evidence that complex models beat RandomWalk."
        ),
    }


def basis_payload(days: int = 365) -> dict:
    frame = load_master().dropna(subset=[TARGET, "london_vnd_kg_lag1"]).tail(max(1, min(days, 2000))).copy()
    frame["basis"] = frame[TARGET] - frame["london_vnd_kg_lag1"]
    return {
        "unit": "VND/kg",
        "data": [
            {
                "date": row.Ngay.date().isoformat(),
                "domestic": round(float(row.Gia_target), 2),
                "london": round(float(row.london_vnd_kg_lag1), 2),
                "basis": round(float(row.basis), 2),
            }
            for row in frame.itertuples()
        ],
    }


def drivers_payload(days: int = 365) -> dict:
    master = load_master().tail(max(1, min(days, 2000))).copy()
    available = {label: column for label, column in DRIVER_COLUMNS.items() if column in master.columns}
    values = master[list(available.values())].astype(float)
    standardised = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    data = []
    for index, row in master.iterrows():
        item = {"date": pd.Timestamp(row["Ngay"]).date().isoformat()}
        for label, column in available.items():
            value = standardised.loc[index, column]
            item[label] = None if pd.isna(value) else round(float(value), 4)
        data.append(item)
    return {"unit": "z-score", "drivers": list(available), "data": data}


def reliability_payload(horizon: int) -> dict:
    comparison = comparison_payload(horizon)
    registry = load_registry()
    deployed = registry[registry["horizon"] == horizon] if not registry.empty else pd.DataFrame()
    deployed_map = {str(row.model): row for row in deployed.itertuples()}
    rows = []
    for item in comparison["models"]:
        metric = deployed_map.get(item["model"])
        rows.append({
            **item,
            "deployed": metric is not None,
            "mae": round(float(metric.MAE), 2) if metric is not None else None,
            "rmse_deployed": round(float(metric.RMSE), 2) if metric is not None else None,
            "directional_accuracy": round(float(metric.DA_pct), 2) if metric is not None else None,
            "mcs_conclusion": "No robust evidence of superiority over RandomWalk",
        })
    return {
        "horizon": horizon,
        "out_of_sample_points": RESEARCH_SAMPLE[horizon],
        "models": rows,
        "mcs": {
            "method": "Model Confidence Set; stationary block bootstrap",
            "conclusion": "Complex models do not beat the RandomWalk benchmark with robust statistical evidence.",
            "provenance": "offline research pipeline in scripts/analysis/12_mcs.py",
        },
    }


def ingestion_status(db: Session) -> dict:
    rows = db.scalars(select(IngestionRun).order_by(IngestionRun.finished_at.desc()).limit(20)).all()
    return {
        "master_data": str(settings.master_data_path),
        "model_registry": str(settings.model_registry_path),
        "runs": [
            {
                "source": row.source,
                "status": row.status,
                "rows_loaded": row.rows_loaded,
                "detail": row.detail,
                "finished_at": row.finished_at.isoformat(),
            }
            for row in rows
        ],
    }
