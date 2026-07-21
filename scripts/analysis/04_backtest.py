# -*- coding: utf-8 -*-
"""
04_backtest.py  --  Gia tri kinh te cua tin hieu huong (§5.3).

Bien tin hieu HUONG (tu logistic, khong look-ahead) thanh mot chien luoc
Long/Flat theo khoi h ngay KHONG chong lap:
  - Tai moi diem quyet dinh d (buoc h ngay): du bao huong h ngay toi.
  - Neu du bao TANG -> nam giu khoi (loi nhuan khoi = log(gia[d+h]/gia[d])).
  - Neu du bao GIAM -> dung ngoai (loi nhuan 0).

Do luong: tong loi nhuan, Sharpe (annual hoa), max drawdown, va kiem dinh
hoan vi (permutation) so voi phan phoi tin hieu ngau nhien.

Xuat: results/backtest.csv, results/backtest_h{H}_equity.csv,
      results/backtest_h{H}_perm.csv
Chay:  python analysis/04_backtest.py --horizons 5 10
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import (load_master, align_frame, TARGET, INITIAL_FRAC,
                     ANN, F18, max_drawdown)

os.makedirs("results/backtest", exist_ok=True)


def make_logit():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=800, solver="lbfgs"))])


def signals_blockwise(df, features, horizon):
    """Sinh tin hieu Long/Flat theo khoi h ngay KHONG chong lap (khong look-ahead).
    Tra ve (dates, block_log_returns_thuc, signals[0/1])."""
    d, have = align_frame(df, features)
    price = d[TARGET].astype(float).values
    X = d[have].astype(float).values
    dates = d["Ngay"].values if "Ngay" in d.columns else np.arange(len(d))
    n = len(d)
    start = int(n * INITIAL_FRAC)
    ds, rets, sigs = [], [], []
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
            block_ret = float(np.log(price[dd + horizon] / price[dd]))
            ds.append(dates[dd]); rets.append(block_ret); sigs.append(sig)
        dd += horizon                       # khoi KHONG chong lap
    return np.array(ds), np.array(rets), np.array(sigs)


def metrics(block_ret, sig, horizon):
    strat = sig * block_ret                 # log-return chien luoc theo khoi
    equity = np.cumprod(np.exp(strat))
    total = float(equity[-1] - 1.0) if len(equity) else float("nan")
    n_blk = len(strat)
    if n_blk > 1 and strat.std(ddof=1) > 0:
        sharpe = float(strat.mean() / strat.std(ddof=1) * np.sqrt(ANN / horizon))
    else:
        sharpe = float("nan")
    mdd = max_drawdown(equity) if n_blk else float("nan")
    return strat, equity, total, sharpe, mdd


def permutation_test(block_ret, sig, horizon, n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    obs = float((sig * block_ret).sum())
    k = int(sig.sum()); n = len(sig)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        s = np.zeros(n, dtype=int)
        s[rng.choice(n, size=k, replace=False)] = 1     # giu nguyen so lan Long
        perm[i] = float((s * block_ret).sum())
    p = float((perm >= obs).mean())
    return obs, perm, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--n_perm", type=int, default=2000)
    a = ap.parse_args()

    df = load_master()
    rows = []
    for h in a.horizons:
        ds, ret, sig = signals_blockwise(df, F18, h)
        if len(ret) == 0:
            print(f"[bo qua] h={h}: khong du du lieu"); continue
        strat, equity, total, sharpe, mdd = metrics(ret, sig, h)
        obs, perm, pval = permutation_test(ret, sig, h, n_perm=a.n_perm)
        rows.append(dict(horizon=h, n_blocks=len(ret), n_long=int(sig.sum()),
                         total_return_pct=round(total * 100, 1),
                         sharpe_ann=round(sharpe, 2), max_drawdown_pct=round(mdd * 100, 1),
                         perm_p=round(pval, 4)))
        pd.DataFrame({"date": ds, "equity": equity, "strat_logret": strat,
                      "signal": sig}).to_csv(f"results/backtest/backtest_h{h}_equity.csv", index=False)
        pd.DataFrame({"perm_sum": perm}).to_csv(f"results/backtest/backtest_h{h}_perm.csv", index=False)
        print(f"h={h:2d}: total={total*100:+.1f}%  Sharpe={sharpe:.2f}  "
              f"MDD={mdd*100:.1f}%  perm_p={pval:.4f}  (n_blocks={len(ret)})")
    out = pd.DataFrame(rows)
    out.to_csv("results/backtest/backtest.csv", index=False)
    print("\n[done] -> results/backtest/backtest.csv (+ equity/perm per horizon)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
