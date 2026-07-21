# -*- coding: utf-8 -*-
"""
02_direction_test.py  --  Kiem dinh KHA NANG DU BAO HUONG (§5.2 / §6).

Cau hoi: sau khi chong look-ahead, mo hinh co doan dung HUONG tang/giam
gia(t+h) so voi gia(t) tot hon ngau nhien khong?

Phuong phap:
  - Nhan: y = 1 neu gia(t+h) > gia(t), nguoc lai 0.
  - Mo hinh: Logistic Regression (L2), chuan hoa dac trung.
    (Xap xi logistic-GD lr0.2/iters800/l2=1e-3 mo ta trong bai; sklearn on dinh hon.)
  - Walk-forward expanding window: tai diem quyet dinh d, chi train tren cac cap
    (t, t+h) da hoan thanh (t+h <= d) -> KHONG nhin tuong lai.
  - So sanh 2 bo dac trung: F18 (chinh thuc) vs F24 (mo rong, them bien cung).
  - Do luong: Directional Accuracy (DA) + kiem dinh Pesaran-Timmermann (1992).

Xuat: results/direction_test.csv
Chay:  python analysis/02_direction_test.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import (load_master, align_frame, TARGET, INITIAL_FRAC, STEP,
                     HORIZONS, F18, F24)

os.makedirs("results/statistical", exist_ok=True)

def make_logit():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, penalty="l2",
                                              max_iter=800, solver="lbfgs"))])


def pesaran_timmermann(y, p):
    """PT (1992) test doc lap huong. y, p la nhan 0/1 (thuc te vs du bao).
    Tra ve (S_stat, p_value 1 phia)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    P = np.mean(y == p)                       # ti le doan dung
    Py = np.mean(y); Px = np.mean(p)
    Pstar = Py * Px + (1 - Py) * (1 - Px)      # ky vong khi doc lap
    var_P = Pstar * (1 - Pstar) / n
    var_Pstar = (((2 * Py - 1) ** 2) * Px * (1 - Px) / n +
                 ((2 * Px - 1) ** 2) * Py * (1 - Py) / n +
                 4 * Py * Px * (1 - Py) * (1 - Px) / (n ** 2))
    denom = var_P - var_Pstar
    if denom <= 0:
        return float("nan"), float("nan")
    S = (P - Pstar) / np.sqrt(denom)
    pval = 1 - norm.cdf(S)                     # 1 phia: doan tot hon ngau nhien
    return float(S), float(pval)


def walk_forward_direction(df, features, horizon):
    d, have = align_frame(df, features)
    price = d[TARGET].astype(float).values
    X = d[have].astype(float).values
    n = len(d)
    start = int(n * INITIAL_FRAC)
    y_true, y_pred = [], []
    for dd in range(start, n - horizon, STEP):
        tr = dd - horizon + 1                  # so dong train hop le (t+h <= dd)
        if tr <= 20:
            continue
        # nhan train: 1 neu gia(t+h) > gia(t), cho t = 0..tr-1 (deu da hoan thanh)
        ytr = (price[horizon:tr + horizon] > price[:tr]).astype(int)
        Xtr = X[:tr]
        if len(np.unique(ytr)) < 2:
            continue
        m = make_logit(); m.fit(Xtr, ytr)
        prob = float(m.predict_proba(X[[dd]])[0, 1])
        y_pred.append(int(prob > 0.5))
        y_true.append(int(price[dd + horizon] > price[dd]))
    return np.array(y_true), np.array(y_pred)


def main():
    df = load_master()
    rows = []
    for tag, feats in [("F18", F18), ("F24", F24)]:
        for h in HORIZONS:
            if h == 1:
                # huong 1 ngay it y nghia kinh te; van tinh de tham khao
                pass
            yt, yp = walk_forward_direction(df, feats, h)
            if len(yt) == 0:
                continue
            da = float(np.mean(yt == yp) * 100)
            S, pv = pesaran_timmermann(yt, yp)
            rows.append(dict(feature_set=tag, h=h, n=len(yt),
                             DA_pct=round(da, 2), PT_stat=round(S, 3), PT_p=round(pv, 5)))
            print(f"{tag} h={h:2d}: n={len(yt):4d}  DA={da:5.2f}%  PT={S:6.3f}  p={pv:.5f}")
    out = pd.DataFrame(rows)
    out.to_csv("results/statistical/direction_test.csv", index=False)
    print("\n[done] -> results/statistical/direction_test.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
