# -*- coding: utf-8 -*-
"""
09_backtest_regime_dca.py  --  Tach REGIME + so voi DCA nong dan (§5.3).

Bo sung cho 04_backtest.py:
  (1) Tach hieu qua chien luoc Long/Flat theo HAI CHE DO thi truong:
        - Regime 2020-2023 (di ngang)
        - Regime 2024-2025 (tang manh)
      Bao cao total_return, Sharpe (annual hoa) cho tung regime.
  (2) So sanh gia ban trung binh THEO TIN HIEU voi chien luoc DCA (ban deu):
        - DCA:    ban 1 don vi moi khoi -> gia ban tb = mean(gia tai diem quyet dinh)
        - Signal: neu du bao TANG -> hoan ban (giu sang khoi sau);
                  neu du bao GIAM -> ban ngay tai gia hien tai.
                  Con ton cuoi ky ban het o gia cuoi.
      Bao cao gia ban tb (nghin VND) cua hai cach.

Dung CUNG khung walk-forward (INITIAL_FRAC=0.6), khoi h ngay KHONG chong lap,
logistic F18, giong 04_backtest.py -> chi bo sung phan bao cao.

Xuat: results/backtest/backtest_regime.csv, results/backtest/backtest_dca.csv
Chay: python analysis/09_backtest_regime_dca.py --horizons 5 10
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import load_master, align_frame, TARGET, INITIAL_FRAC, ANN, F18

os.makedirs("results/backtest", exist_ok=True)
REGIME_SPLIT = pd.Timestamp("2024-01-01")     # ranh gioi hai che do


def make_logit():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=800, solver="lbfgs"))])


def signals_blockwise(df, features, horizon):
    d, have = align_frame(df, features)
    price = d[TARGET].astype(float).values
    X = d[have].astype(float).values
    dates = pd.to_datetime(d["Ngay"]).values if "Ngay" in d.columns else np.arange(len(d))
    n = len(d)
    start = int(n * INITIAL_FRAC)
    ds, px, rets, sigs = [], [], [], []
    dd = start
    while dd < n - horizon:
        tr = dd - horizon + 1
        if tr > 20:
            ytr = (price[horizon:tr + horizon] > price[:tr]).astype(int)
            if len(np.unique(ytr)) >= 2:
                m = make_logit(); m.fit(X[:tr], ytr)
                sig = int(m.predict_proba(X[[dd]])[0, 1] > 0.5)
            else:
                sig = 0
            ds.append(dates[dd]); px.append(price[dd])
            rets.append(float(np.log(price[dd + horizon] / price[dd]))); sigs.append(sig)
        dd += horizon
    return (np.array(ds), np.array(px, float), np.array(rets, float), np.array(sigs, int))


def regime_metrics(strat, horizon):
    if len(strat) < 2 or strat.std(ddof=1) == 0:
        return float("nan"), float("nan")
    total = float(np.prod(np.exp(strat)) - 1.0)
    sharpe = float(strat.mean() / strat.std(ddof=1) * np.sqrt(ANN / horizon))
    return total, sharpe


def dca_vs_signal(px, sig):
    """Gia ban tb (nghin VND): DCA (ban deu) vs Signal (hoan ban khi du bao tang)."""
    dca_avg = float(np.mean(px)) / 1000.0
    qty = 0.0; revenue = 0.0
    for i in range(len(px)):
        qty += 1.0                       # moi khoi phat sinh 1 don vi can ban
        if sig[i] == 0:                  # du bao GIAM -> ban het ton hien tai
            revenue += qty * px[i]; qty = 0.0
    if qty > 0:                          # con ton -> ban o gia cuoi
        revenue += qty * px[-1]
    signal_avg = (revenue / len(px)) / 1000.0
    return dca_avg, signal_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10])
    a = ap.parse_args()

    df = load_master()
    reg_rows, dca_rows = [], []
    for h in a.horizons:
        ds, px, ret, sig = signals_blockwise(df, F18, h)
        if len(ret) == 0:
            print(f"[bo qua] h={h}"); continue
        strat = sig * ret
        dts = pd.to_datetime(ds)
        for label, mask in [("2020-2023", dts < REGIME_SPLIT),
                            ("2024-2025", dts >= REGIME_SPLIT)]:
            tot, shp = regime_metrics(strat[mask], h)
            reg_rows.append(dict(horizon=h, regime=label, n_blocks=int(mask.sum()),
                                 total_return_pct=round(tot * 100, 1) if tot == tot else np.nan,
                                 sharpe_ann=round(shp, 2) if shp == shp else np.nan))
            print(f"h={h:2d} {label}: n={int(mask.sum()):3d}  total={tot*100:+.1f}%  Sharpe={shp:.2f}")
        dca_avg, sig_avg = dca_vs_signal(px, sig)
        dca_rows.append(dict(horizon=h, dca_avg_k=round(dca_avg, 1), signal_avg_k=round(sig_avg, 1),
                             diff_k=round(sig_avg - dca_avg, 1)))
        print(f"h={h:2d} DCA={dca_avg:.1f}k  Signal={sig_avg:.1f}k (nghin VND)")
    pd.DataFrame(reg_rows).to_csv("results/backtest/backtest_regime.csv", index=False)
    pd.DataFrame(dca_rows).to_csv("results/backtest/backtest_dca.csv", index=False)
    print("\n[done] -> results/backtest/backtest_regime.csv + backtest_dca.csv")


if __name__ == "__main__":
    main()
