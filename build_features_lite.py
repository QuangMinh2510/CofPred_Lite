#!/usr/bin/env python3
"""Tạo bộ feature gọn từ giá tỉnh, London, dầu diesel và thời tiết."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
COFFEE_PATH = ROOT / "data" / "Processing" / "coffe" / "gia_cafe.csv"
LONDON_PATH = ROOT / "data" / "Processing" / "coffe" / "london_robusta_price.csv"
FUEL_PATH = ROOT / "data" / "Processing" / "Fuel" / "diesel_2017_now_daily.csv"
WEATHER_PATH = ROOT / "data" / "Raw" / "weather" / "weather_all_regions.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "gia_cafe_features_lite.csv"


def build_features() -> pd.DataFrame:
    coffee = pd.read_csv(COFFEE_PATH)
    coffee["Ngay"] = pd.to_datetime(coffee["Ngay"], errors="coerce")
    coffee["Gia"] = pd.to_numeric(coffee["Gia"], errors="coerce")
    coffee = (
        coffee.dropna(subset=["Ngay", "Gia"])
        .groupby("Ngay", as_index=False)["Gia"].mean()
        .rename(columns={"Gia": "Gia_target"})
    )

    london = pd.read_csv(LONDON_PATH)
    london["Date"] = pd.to_datetime(london["Date"], errors="coerce")
    london["Price"] = pd.to_numeric(london["Price"], errors="coerce")
    london = london[["Date", "Price"]].rename(
        columns={"Date": "Ngay", "Price": "london_usd_ton"}
    )

    fuel = pd.read_csv(FUEL_PATH)
    fuel["ngay"] = pd.to_datetime(fuel["ngay"], errors="coerce")
    fuel["gia"] = pd.to_numeric(fuel["gia"], errors="coerce")
    fuel = fuel[["ngay", "gia"]].rename(columns={"ngay": "Ngay", "gia": "diesel"})

    weather = pd.read_csv(WEATHER_PATH)
    weather["date"] = (
        pd.to_datetime(weather["date"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Bangkok")
        .dt.tz_localize(None)
    )
    weather["date"] = weather["date"].dt.normalize()
    for column in ["rain_sum", "temperature_2m_mean", "relative_humidity_2m_mean"]:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    weather = weather.groupby("date", as_index=False).agg(
        rain=("rain_sum", "mean"),
        temperature_mean=("temperature_2m_mean", "mean"),
        humidity_mean=("relative_humidity_2m_mean", "mean"),
    ).rename(columns={"date": "Ngay"})

    frame = coffee.merge(london, on="Ngay", how="left")
    frame = frame.merge(fuel, on="Ngay", how="left")
    frame = frame.merge(weather, on="Ngay", how="left").sort_values("Ngay")
    frame[["london_usd_ton", "diesel"]] = frame[["london_usd_ton", "diesel"]].ffill()

    shifted_price = frame["Gia_target"].shift(1)
    frame["target_lag1"] = frame["Gia_target"].shift(1)
    frame["target_lag2"] = frame["Gia_target"].shift(2)
    frame["target_lag3"] = frame["Gia_target"].shift(3)
    frame["target_ret_lag1"] = frame["Gia_target"].pct_change()
    frame["MA5"] = shifted_price.rolling(5).mean()
    frame["MA10"] = shifted_price.rolling(10).mean()
    frame["std5"] = shifted_price.rolling(5).std()
    frame["dayofweek"] = frame["Ngay"].dt.dayofweek
    frame["month"] = frame["Ngay"].dt.month
    frame["london_usd_ton_lag1"] = frame["london_usd_ton"].shift(1)
    frame["diesel_chg_1m"] = frame["diesel"].pct_change(21)
    frame["diesel_chg_3m"] = frame["diesel"].pct_change(63)
    frame["rain_90d"] = frame["rain"].rolling(90, min_periods=20).sum()
    return frame.reset_index(drop=True)


if __name__ == "__main__":
    result = build_features()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã tạo {len(result):,} dòng và {len(result.columns)} cột: {OUTPUT_PATH}")
