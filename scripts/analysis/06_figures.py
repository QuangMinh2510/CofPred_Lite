# -*- coding: utf-8 -*-
"""
06_figures.py  --  Sinh 3 hinh minh hoa cho bai bao / luan an.

  Hinh 1: gia Robusta noi dia 2020-2025 + moc gay cau truc (Gregory-Hansen).
  Hinh 2: equity curve chien luoc (tu 04_backtest.py) vs Buy&Hold.
  Hinh 3: phan phoi permutation test (tu 04_backtest.py) + gia tri quan sat.

Uu tien doc ket qua that tu results/ (do 04_backtest & 05 sinh ra). Neu chua co,
hinh 2/3 se bao thieu va huong dan chay truoc.

Xuat: results/fig1_price.png, results/fig2_equity.png, results/fig3_permutation.png
Chay:  python analysis/06_figures.py --break_date 2023-07-04
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import load_master, TARGET

os.makedirs("results/statistical", exist_ok=True)


def fig1_price(break_date):
    df = load_master().dropna(subset=[TARGET])
    x = df["Ngay"] if "Ngay" in df.columns else np.arange(len(df))
    plt.figure(figsize=(9, 4))
    plt.plot(x, df[TARGET], color="#1f77b4", lw=1.1)
    if break_date:
        plt.axvline(pd.Timestamp(break_date), ls="--", color="red",
                    label=f"Gay cau truc ({break_date})")
        plt.legend()
    plt.title("Gia Robusta noi dia 2020-2025"); plt.ylabel("VND/kg")
    plt.tight_layout(); plt.savefig("results/statistical/fig1_price.png", dpi=150); plt.close()
    print("saved results/statistical/fig1_price.png")


def fig2_equity(horizon):
    f = f"results/backtest/backtest_h{horizon}_equity.csv"
    if not os.path.exists(f):
        print(f"[thieu] {f} -> chay 04_backtest.py --horizons {horizon} truoc"); return
    d = pd.read_csv(f, parse_dates=["date"], dayfirst=False)
    plt.figure(figsize=(9, 4))
    plt.plot(d["date"], d["equity"], color="#e84393", lw=1.6, label="Chien luoc (Long/Flat)")
    plt.axhline(1, ls=":", color="black", lw=0.8)
    plt.title(f"Equity Curve - chien luoc theo tin hieu huong (h={horizon})")
    plt.ylabel("Tang truong von (x lan)"); plt.legend()
    plt.tight_layout(); plt.savefig("results/statistical/fig2_equity.png", dpi=150); plt.close()
    print("saved results/statistical/fig2_equity.png")


def fig3_perm(horizon):
    f = f"results/backtest/backtest_h{horizon}_perm.csv"
    ef = f"results/backtest/backtest_h{horizon}_equity.csv"
    if not (os.path.exists(f) and os.path.exists(ef)):
        print(f"[thieu] {f} -> chay 04_backtest.py --horizons {horizon} truoc"); return
    perm = pd.read_csv(f)["perm_sum"].values
    d = pd.read_csv(ef)
    observed = float(d["strat_logret"].sum())
    pval = (perm >= observed).mean()
    plt.figure(figsize=(8, 4))
    plt.hist(perm, bins=50, color="#5b8dd9", edgecolor="white", alpha=0.85, label="Ngau nhien")
    plt.axvline(observed, color="red", lw=2, label=f"Quan sat = {observed:.4f}")
    plt.title(f"Permutation Test (h={horizon}) - p = {pval:.3f}")
    plt.xlabel("Tong log-return"); plt.ylabel("Tan so"); plt.legend()
    plt.tight_layout(); plt.savefig("results/statistical/fig3_permutation.png", dpi=150); plt.close()
    print("saved results/statistical/fig3_permutation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--break_date", default="2023-07-04")
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()
    fig1_price(a.break_date)
    fig2_equity(a.horizon)
    fig3_perm(a.horizon)
    print("\n[done] -> results/statistical/fig1_price.png, results/statistical/fig2_equity.png, results/statistical/fig3_permutation.png")


if __name__ == "__main__":
    main()
