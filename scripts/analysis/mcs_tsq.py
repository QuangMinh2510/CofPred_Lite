#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcs_tsq.py -- Model Confidence Set with the SEMI-QUADRATIC statistic T_SQ
=========================================================================
Companion to 12_mcs.py (which uses the range statistic T_R). Hansen, Lunde &
Nason (2011) propose two asymptotically valid equivalence-test statistics for
the MCS:

    range statistic          T_R  = max_{i,j in M} |t_ij|
    semi-quadratic statistic T_SQ = sum_{i<j in M} t_ij^2

with t_ij = dbar_ij / sqrt( var*(dbar_ij) ) the standardized pairwise mean-loss
difference. BOTH statistics share the SAME elimination rule (drop the model
with the largest standardized excess loss t_{i.}); they differ ONLY in how the
surviving set is tested for equal predictive accuracy. Reporting T_SQ next to
the T_R results of 12_mcs.py is the robustness check the referee asked for.

This script reuses EXACTLY the same inputs, date alignment, loss construction,
seed and Politis-Romano stationary block bootstrap as 12_mcs.py, so the two
runs are directly comparable. Depends on numpy/pandas only.

RUN (identical interface to 12_mcs.py):
  python mcs_tsq.py --preds results/preds --h 5  --loss mae --alpha 0.10
  python mcs_tsq.py --preds results/preds --h 5  --loss mse --block 10 --B 3000
  python mcs_tsq.py --preds results/preds --h 1  --loss mae
  python mcs_tsq.py --preds results/preds --h 21 --loss mse
  python mcs_tsq.py --preds results/preds --h 63 --loss mae

OUTPUT: results/robustness/mcs_tsq_h<h>_<loss>.csv  (same schema as mcs_h*.csv:
        model, status, p_MCS, MAE_or_MSE). Compare 'status' / 'p_MCS' column
        against the matching mcs_h<h>_<loss>.csv produced by 12_mcs.py.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd


def load_losses(preds_dir, h, loss='mae'):
    """Identical to 12_mcs.py: read every <MODEL>__h<h>.csv (cols Ngay,y_true,
    pred), align on the common set of dates, and return (models, dates, L) with
    L of shape (T, M) holding the per-point loss of each model."""
    files = sorted(glob.glob(os.path.join(preds_dir, f'*__h{h}.csv')))
    if not files:
        raise SystemExit(f'Khong tim thay file *__h{h}.csv trong {preds_dir}')
    series = {}
    for f in files:
        name = re.sub(r'__h\d+\.csv$', '', os.path.basename(f))
        d = pd.read_csv(f)
        d['Ngay'] = pd.to_datetime(d['Ngay'])
        err = (d['y_true'].astype(float) - d['pred'].astype(float))
        l = err.abs() if loss == 'mae' else err**2
        series[name] = pd.Series(l.values, index=d['Ngay'].values)
    L = pd.DataFrame(series).dropna(axis=0, how='any')  # giao nhau theo Ngay
    L = L.sort_index()
    return list(L.columns), L.index.values, L.values


def stationary_bootstrap_idx(T, B, block, rng):
    """Politis-Romano stationary bootstrap: return an (B, T) index array.
    Byte-for-byte the same routine as 12_mcs.py."""
    p = 1.0 / max(block, 1)
    idx = np.empty((B, T), dtype=np.int64)
    for b in range(B):
        cur = rng.integers(0, T)
        for t in range(T):
            if t == 0 or rng.random() < p:
                cur = rng.integers(0, T)
            else:
                cur = (cur + 1) % T
            idx[b, t] = cur
    return idx


def _sumsq_upper(tmat, k):
    """Sum of squares over the strict upper triangle (i<j) of a (k,k) matrix.
    This is the semi-quadratic aggregation that defines T_SQ."""
    iu = np.triu_indices(k, k=1)
    return float(np.sum(np.square(tmat[iu])))


def mcs_tsq(L, alpha=0.10, B=3000, block=10, seed=1234):
    """MCS using the semi-quadratic statistic T_SQ. Uses the same seed and
    bootstrap draws as 12_mcs.py's mcs(), so the elimination path matches and
    only the equivalence-test p-value can differ. Returns a list of dicts."""
    T, M = L.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(T, B, block, rng)
    bmean = L[idx].mean(axis=1)          # (B, M)
    full  = L.mean(axis=0)               # (M,)
    z = bmean - full                     # (B, M): bootstrap deviations about the mean

    included = list(range(M))
    elim_order, elim_p = [], []
    running = 0.0
    while len(included) > 1:
        sub = np.array(included); k = len(sub)
        Lm = full[sub]                   # (k,)
        zk = z[:, sub]                   # (B, k)
        # bootstrap variance of each pairwise difference d_ij = Lm_i - Lm_j
        var_ij = np.zeros((k, k))
        for i in range(k):
            diff = zk[:, i][:, None] - zk      # (B, k)
            var_ij[i] = diff.var(axis=0)
        dij = Lm[:, None] - Lm[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            tij = np.where(var_ij > 0, dij / np.sqrt(var_ij), 0.0)
        # --- SEMI-QUADRATIC equivalence statistic (this is the only change vs T_R) ---
        TSQ = _sumsq_upper(tij, k)
        TSQ_b = np.empty(B)
        for b in range(B):
            zb = zk[b]
            db = zb[:, None] - zb[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                tb = np.where(var_ij > 0, db / np.sqrt(var_ij), 0.0)
            TSQ_b[b] = _sumsq_upper(tb, k)
        p = float(np.mean(TSQ_b >= TSQ))
        running = max(running, p)         # p_MCS = cumulative max
        if p >= alpha:
            break                         # every remaining model is in the MCS
        # elimination rule: identical to 12_mcs.py (largest standardized excess loss)
        di = Lm - Lm.mean()
        vi = (zk - zk.mean(axis=1, keepdims=True)).var(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ti = np.where(vi > 0, di / np.sqrt(vi), 0.0)
        worst = int(np.argmax(ti))
        elim_order.append(sub[worst]); elim_p.append(running)
        included.remove(sub[worst])

    rows = []
    for m, pv in zip(elim_order, elim_p):
        rows.append(dict(model_idx=m, status='eliminated', p_MCS=round(pv, 4)))
    for m in included:
        rows.append(dict(model_idx=m, status='IN_MCS', p_MCS=round(max(running, alpha), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='results/preds')
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--loss', choices=['mae','mse'], default='mae')
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--B', type=int, default=3000)
    ap.add_argument('--block', type=int, default=10)
    ap.add_argument('--out', default='results/robustness')
    a = ap.parse_args()

    models, dates, L = load_losses(a.preds, a.h, a.loss)
    print(f'[MCS-TSQ] h={a.h} loss={a.loss} | {len(models)} model | T={L.shape[0]} diem chung '
          f'({pd.Timestamp(dates.min()).date()} -> {pd.Timestamp(dates.max()).date()})')
    rows = mcs_tsq(L, alpha=a.alpha, B=a.B, block=a.block)
    df = pd.DataFrame(rows)
    df['model'] = df['model_idx'].map(lambda i: models[i])
    df['MAE_or_MSE'] = df['model_idx'].map(lambda i: round(float(L[:, i].mean()), 2))
    df = df[['model','status','p_MCS','MAE_or_MSE']].sort_values(
        ['status','MAE_or_MSE'], ascending=[True, True]).reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, f'mcs_tsq_h{a.h}_{a.loss}.csv')
    df.to_csv(fp, index=False)
    print('\n=== KET QUA MCS / T_SQ (alpha={:.2f}) ==='.format(a.alpha))
    print(df.to_string(index=False))
    print(f'\nModel nam trong MCS theo T_SQ (muc tin cay {1-a.alpha:.0%}):',
          ', '.join(df.loc[df.status=='IN_MCS','model']))
    print(f'-> luu: {fp}')


if __name__ == '__main__':
    main()
