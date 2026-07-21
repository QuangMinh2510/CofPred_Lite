# -*- coding: utf-8 -*-
"""
07_variance_ratio.py  --  Kiem dinh TI SO PHUONG SAI Lo-MacKinlay (1988) (§5.1).

Muc dich: kiem dinh gia thuyet random walk tren MUC GIA. Neu gia tuan theo
random walk thi phuong sai cua log-return q-ky phai tang tuyen tinh theo q,
nen ti so phuong sai VR(q) = Var(q-period)/(q*Var(1-period)) ~ 1.

Bao cao ca sai so chuan dong nhat (homoskedastic) va vung voi phuong sai thay
doi (heteroskedasticity-robust) theo Lo & MacKinlay (1988). Dung CUNG mau
N=1516 (align theo F18) de nhat quan voi cac bang khac.

Xuat: results/statistical/variance_ratio.csv
Chay:  python analysis/07_variance_ratio.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import load_master, align_frame, TARGET, F18

os.makedirs("results/statistical", exist_ok=True)


def variance_ratio(x, q):
    """VR(q) + z-stat (homo & hetero-robust) theo Lo-MacKinlay (1988)."""
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / (n - 1)
    m = q * (n - q + 1) * (1.0 - q / n)          # hieu chinh khong chech
    xq = np.array([x[i:i + q].sum() for i in range(n - q + 1)])
    varq = np.sum((xq - q * mu) ** 2) / m
    vr = varq / (q * var1)
    # --- homoskedastic ---
    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)
    z_homo = (vr - 1.0) / np.sqrt(phi) if phi > 0 else float("nan")
    p_homo = 2 * (1 - norm.cdf(abs(z_homo)))
    # --- heteroskedasticity-robust ---
    den = (np.sum((x - mu) ** 2)) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum(((x[j:] - mu) ** 2) * ((x[:-j] - mu) ** 2))
        delta_j = n * num / den
        theta += ((2.0 * (q - j) / q) ** 2) * delta_j
    z_het = (vr - 1.0) / np.sqrt(theta) if theta > 0 else float("nan")
    p_het = 2 * (1 - norm.cdf(abs(z_het)))
    return vr, z_homo, p_homo, z_het, p_het


def main():
    df = load_master()
    d, have = align_frame(df, F18)               # cung mau N=1516
    price = d[TARGET].astype(float).values
    logret = np.diff(np.log(price))
    rows = []
    for q in [2, 5, 10, 21]:
        vr, zho, pho, zhe, phe = variance_ratio(logret, q)
        rows.append(dict(q=q, VR=round(vr, 4),
                         z_homo=round(zho, 3), p_homo=round(pho, 4),
                         z_hetero=round(zhe, 3), p_hetero=round(phe, 4)))
        print(f"q={q:2d}: VR={vr:.4f}  z_homo={zho:+.3f}(p={pho:.4f})  "
              f"z_hetero={zhe:+.3f}(p={phe:.4f})")
    out = pd.DataFrame(rows)
    out.to_csv("results/statistical/variance_ratio.csv", index=False)
    print("\n[done] -> results/statistical/variance_ratio.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
