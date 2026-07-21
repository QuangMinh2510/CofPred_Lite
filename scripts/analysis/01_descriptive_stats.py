# -*- coding: utf-8 -*-
"""
01_descriptive_stats.py  --  Thong ke mo ta + kiem dinh tinh dung + tuong quan.

TACH tu thongke.py cu (chi giu phan SO LIEU, bo phan ve hinh -> xem 06_figures.py).
SUA: dung ten cot dac trung DUNG voi master (london_vnd_kg_lag1, usdvnd_lag1, ...)
     thay vi ten tho de tranh bi bo qua thau lang.

Xuat: results/descriptive_stats.csv, results/stationarity.csv, in tuong quan ra man hinh.
Chay:  python analysis/01_descriptive_stats.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import load_master, TARGET

os.makedirs("results/statistical", exist_ok=True)

# Bien ngoai sinh mo ta (ten DUNG voi master; contemporaneous neu co, else lag1)
EXO_DESC = ["london_vnd_kg", "usdvnd", "diesel", "oni", "rain_90d", "waterbal_90d"]


def _resolve(df, name):
    if name in df.columns:
        return name
    for alt in (f"{name}_lag1", f"{name}_lag1m"):
        if alt in df.columns:
            return alt
    return None


def desc(x):
    x = pd.Series(x).dropna().astype(float)
    jb, jbp = jarque_bera(x)[:2]
    return dict(N=len(x), mean=x.mean(), sd=x.std(), min=x.min(),
                p25=x.quantile(.25), median=x.median(), p75=x.quantile(.75),
                max=x.max(), skew=skew(x), kurt=kurtosis(x, fisher=False),
                JB=jb, JB_p=jbp)


def main():
    df = load_master()
    df = df.dropna(subset=[TARGET]).copy()
    df["ret"] = np.log(df[TARGET]).diff()

    # --- Thong ke mo ta ---
    rows = []
    series = [("Gia (level)", df[TARGET]), ("Log-return", df["ret"])]
    for name in EXO_DESC:
        col = _resolve(df, name)
        if col is not None:
            series.append((col, df[col]))
        else:
            print(f"[bo qua] khong tim thay cot cho '{name}'")
    print("=== DESCRIPTIVE STATS ===")
    for name, s in series:
        d = desc(s)
        d = {"var": name, **{k: round(v, 4) for k, v in d.items()}}
        rows.append(d)
        print(name, {k: v for k, v in d.items() if k != "var"})
    pd.DataFrame(rows).to_csv("results/statistical/descriptive_stats.csv", index=False)

    # --- Tinh dung ADF + KPSS ---
    print("\n=== STATIONARITY ===")
    st_rows = []
    for name, s in [("Gia (level)", df[TARGET].dropna()),
                    ("Log-return", df["ret"].dropna())]:
        adf = adfuller(s, autolag="AIC")
        kp = kpss(s, regression="c", nlags="auto")
        st_rows.append(dict(series=name, ADF_stat=round(adf[0], 3), ADF_p=round(adf[1], 4),
                            KPSS_stat=round(kp[0], 3), KPSS_p=round(kp[1], 4)))
        print(f"{name}: ADF stat={adf[0]:.3f} p={adf[1]:.4f} | "
              f"KPSS stat={kp[0]:.3f} p={kp[1]:.4f}")
    pd.DataFrame(st_rows).to_csv("results/statistical/stationarity.csv", index=False)
    print("  (KPSS p la CAN: gia tri in ra co the bi chan o 0.01/0.10.)")

    # --- Tuong quan ret vs EXO ---
    print("\n=== CORRELATION (ret vs EXO) ===")
    cols = ["ret"] + [c for c in (_resolve(df, n) for n in EXO_DESC) if c]
    print(df[cols].corr().round(3).to_string())

    print("\n[done] -> results/statistical/descriptive_stats.csv , results/statistical/stationarity.csv")


if __name__ == "__main__":
    main()
