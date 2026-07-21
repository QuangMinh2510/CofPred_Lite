#!/usr/bin/env python3
"""Lấy và lưu một ảnh chụp dữ liệu live của CofPred theo ngày."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from live_data import fetch_live_coffee, fetch_live_fuel, fetch_live_weather


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "live"
HISTORY_PATH = OUTPUT_DIR / "daily_snapshots.csv"
LATEST_PATH = OUTPUT_DIR / "latest_snapshot.json"
BUON_MA_THUOT = (12.688928, 108.016312, "Đắk Lắk — Buôn Ma Thuột")


def collect_snapshot() -> dict:
    with ThreadPoolExecutor(max_workers=3) as executor:
        coffee_future = executor.submit(fetch_live_coffee)
        fuel_future = executor.submit(fetch_live_fuel)
        weather_future = executor.submit(fetch_live_weather, *BUON_MA_THUOT)
        return {
            "coffee": coffee_future.result(),
            "fuel": fuel_future.result(),
            "weather": weather_future.result(),
        }


def flatten_snapshot(snapshot: dict) -> dict:
    now = datetime.now()
    coffee = snapshot["coffee"]
    fuel = snapshot["fuel"]
    weather = snapshot["weather"]
    return {
        "collected_at": now.isoformat(timespec="seconds"),
        "collected_date": now.date().isoformat(),
        "coffee_ok": coffee.get("ok", False),
        "coffee_vnd_kg": coffee.get("value"),
        "coffee_change_vnd_kg": coffee.get("delta"),
        "coffee_observed_at": coffee.get("observed_at"),
        "coffee_source": coffee.get("source"),
        "coffee_error": coffee.get("error"),
        "fuel_ok": fuel.get("ok", False),
        "diesel_vnd_litre": fuel.get("value"),
        "diesel_change_vnd_litre": fuel.get("delta"),
        "fuel_observed_at": fuel.get("observed_at"),
        "fuel_source": fuel.get("source"),
        "fuel_error": fuel.get("error"),
        "weather_ok": weather.get("ok", False),
        "temperature_c": weather.get("temperature"),
        "humidity_pct": weather.get("humidity"),
        "rain_mm": weather.get("rain"),
        "wind_kmh": weather.get("wind_speed"),
        "weather_observed_at": weather.get("observed_at"),
        "weather_source": weather.get("source"),
        "weather_error": weather.get("error"),
    }


def save_snapshot(snapshot: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    new_row = pd.DataFrame([flatten_snapshot(snapshot)])
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row
    history = history.drop_duplicates("collected_date", keep="last")
    history.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    current = collect_snapshot()
    save_snapshot(current)
    print(f"Đã lưu snapshot mới: {LATEST_PATH}")
    print(f"Lịch sử theo ngày: {HISTORY_PATH}")
