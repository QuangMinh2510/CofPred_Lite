# -*- coding: utf-8 -*-
"""
05_cointegration_transmission.py  --  Truyen dan gia & khi hau (§5.5).

Gom:
  (A) Engle-Granger: log(gia VN) ~ log(gia London quy VND) -> do co gian dai han
      + ADF phan du (kiem tra dong lien ket).
  (B) ECM: toc do dieu chinh alpha ve can bang dai han.
  (C) Johansen + VECM (kiem tra & uoc luong he dong lien ket).
  (D) Gregory-Hansen: do gay cau truc trong quan he dong lien ket (quet break).
  (E) Granger causality: ONI -> log-return.
  (F) Hoi quy tuong tac ENSO: ret ~ oni + ElNino + oni*ElNino (Newey-West SE).

Xuat: results/cointegration_summary.txt (+ in ra man hinh)
Chay:  python analysis/05_cointegration_transmission.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import load_master, TARGET

os.makedirs("results/statistical", exist_ok=True)
LOG = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def resolve(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def engle_granger(logvn, loglon):
    X = sm.add_constant(loglon)
    r = sm.OLS(logvn, X).fit()
    beta = r.params[1]; tb = r.tvalues[1]
    resid = r.resid
    adf = adfuller(resid, autolag="AIC")
    out(f"(A) Engle-Granger: do co gian dai han beta = {beta:.4f} (t = {tb:.2f})")
    out(f"    ADF phan du: stat = {adf[0]:.3f}, p = {adf[1]:.4f} "
        f"-> {'CO dong lien ket' if adf[1] < 0.05 else 'CHUA ro dong lien ket'}")
    return resid, beta


def ecm(logvn, loglon, resid):
    dvn = logvn.diff(); dlon = loglon.diff(); ect = resid.shift(1)
    d = pd.concat([dvn, dlon, ect], axis=1).dropna()
    d.columns = ["dvn", "dlon", "ect"]
    X = sm.add_constant(d[["ect", "dlon"]])
    r = sm.OLS(d["dvn"], X).fit()
    alpha = r.params["ect"]; ta = r.tvalues["ect"]
    out(f"(B) ECM: toc do dieu chinh alpha = {alpha:.4f} (t = {ta:.2f}) "
        f"-> {'hoi tu ve can bang' if alpha < 0 else 'khong hoi tu'}")


def johansen_vecm(logvn, loglon):
    Y = pd.concat([logvn, loglon], axis=1).dropna()
    Y.columns = ["logvn", "loglon"]
    try:
        jr = coint_johansen(Y.values, det_order=0, k_ar_diff=1)
        out(f"(C) Johansen trace stat = {jr.lr1[0]:.2f} "
            f"(crit 5% = {jr.cvt[0,1]:.2f}) "
            f"-> {'co dong lien ket' if jr.lr1[0] > jr.cvt[0,1] else 'khong'}")
        vecm = VECM(Y.values, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
        out(f"    VECM alpha (toc do dieu chinh) = {np.round(vecm.alpha.ravel(), 4)}")
    except Exception as e:
        out(f"(C) Johansen/VECM loi: {type(e).__name__}: {e}")


def gregory_hansen(logvn, loglon, trim=0.15):
    """Gregory-Hansen (1996) mo hinh C/S (level shift): quet break, chon ADF nho nhat."""
    Y = pd.concat([logvn, loglon], axis=1).dropna().reset_index(drop=True)
    Y.columns = ["logvn", "loglon"]
    n = len(Y)
    lo, hi = int(n * trim), int(n * (1 - trim))
    best = (np.inf, None)
    for b in range(lo, hi):
        dummy = np.zeros(n); dummy[b:] = 1.0
        X = sm.add_constant(pd.DataFrame({"loglon": Y["loglon"], "shift": dummy}))
        res = sm.OLS(Y["logvn"], X).fit().resid
        stat = adfuller(res, autolag="AIC")[0]
        if stat < best[0]:
            best = (stat, b)
    out(f"(D) Gregory-Hansen: ADF* nho nhat = {best[0]:.3f} tai chi so {best[1]}")
    return best[1]


def granger_enso(df):
    oni = resolve(df, "oni")
    if oni is None:
        out("(E)(F) khong tim thay cot 'oni' -> bo qua"); return
    d = df.dropna(subset=[TARGET, oni]).copy()
    d["ret"] = np.log(d[TARGET]).diff()
    d = d.dropna(subset=["ret", oni])
    try:
        gc = grangercausalitytests(d[["ret", oni]], maxlag=5, verbose=False)
        p1 = gc[1][0]["ssr_ftest"][1]
        out(f"(E) Granger ONI -> ret (lag1): p = {p1:.4f} "
            f"-> {'co' if p1 < 0.05 else 'khong'} nhan qua Granger")
    except Exception as e:
        out(f"(E) Granger loi: {type(e).__name__}: {e}")
    # (F) Tuong tac ENSO (ElNino = oni >= 0.5)
    d["elnino"] = (d[oni] >= 0.5).astype(int)
    d["inter"] = d[oni] * d["elnino"]
    X = sm.add_constant(d[[oni, "elnino", "inter"]])
    r = sm.OLS(d["ret"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    bi = r.params["inter"]; pi = r.pvalues["inter"]
    out(f"(F) Tuong tac ONI x ElNino: he so = {bi:.4f} (p = {pi:.4f}) "
        f"[Newey-West HAC, lag 5]")
    out(f"    So ngay ElNino (oni>=0.5): {int(d['elnino'].sum())}")


def main():
    df = load_master().dropna(subset=[TARGET]).copy()
    lon = resolve(df, "london_vnd_kg", "london_vnd_kg_lag1")
    if lon is None:
        out("[loi] khong tim thay cot gia London (london_vnd_kg / _lag1).")
    else:
        d = df.dropna(subset=[TARGET, lon]).copy()
        logvn = np.log(d[TARGET]); loglon = np.log(d[lon])
        resid, _ = engle_granger(logvn, loglon)
        ecm(logvn, loglon, resid)
        johansen_vecm(logvn, loglon)
        gregory_hansen(logvn.reset_index(drop=True), loglon.reset_index(drop=True))
    granger_enso(df)
    with open("results/statistical/cointegration_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    out("\n[done] -> results/statistical/cointegration_summary.txt")


if __name__ == "__main__":
    main()
