#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
cw_spa_tests.py -- Clark-West, Giacomini-White, and SPA/StepM tests
===================================================================
Benchmark = Naive (random walk). Uses ONLY the saved per-date forecasts in
results/preds/<MODEL>__h<h>.csv (columns: Ngay,y_true,pred[,anchor]); NO model
is retrained, so numbers stay perfectly consistent with the paper MCS/DM tables.

Referee points addressed:
  3.1  DM is non-standard for nested pairs (RW nested in ARIMA/ARIMAX/ML-with-
       price-lags). We add Clark-West (2007) MSPE-adjusted test and the
       Giacomini-White (2006) conditional predictive-ability test.
  3.2  SPA test of Hansen (2005) with the random walk as benchmark, answering
       directly: does ANY model beat the RW after controlling data-snooping?
       Plus Romano-Wolf StepM (which models beat the RW, FWER-controlled).
  medium (block length): SPA p-values are reported over several mean block
       lengths as a bootstrap-sensitivity check.

Run (cofpred env; needs numpy/pandas/scipy):
  python cw_spa_tests.py --preds results/preds --h 1 5 21 63 --loss mse
  python cw_spa_tests.py --preds results/preds --h 1 5 21 63 --loss mae

Outputs per horizon:
  results/robustness/cw_gw_h{h}_{loss}.csv   (Clark-West + Giacomini-White vs benchmark)
  results/robustness/spa_h{h}_{loss}.csv     (SPA p-value across block lengths)
'''
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy import stats


def load_preds(preds_dir, h):
    files = sorted(glob.glob(os.path.join(preds_dir, '*__h' + str(h) + '.csv')))
    if not files:
        raise SystemExit('No *__h' + str(h) + '.csv in ' + preds_dir)
    series, ys = {}, {}
    for f in files:
        name = os.path.basename(f).split('__h')[0]
        d = pd.read_csv(f)
        d['Ngay'] = pd.to_datetime(d['Ngay'])
        d = d.sort_values('Ngay')
        series[name] = pd.Series(d['pred'].astype(float).values, index=d['Ngay'].values)
        ys[name] = pd.Series(d['y_true'].astype(float).values, index=d['Ngay'].values)
    preds = pd.DataFrame(series)
    ytrue = pd.DataFrame(ys).mean(axis=1)  # identical across models on common grid
    both = preds.join(ytrue.rename('__y__'), how='inner').dropna()
    y = both.pop('__y__')
    return both.sort_index(), y.sort_index()


def nw_lrv(x, lag):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    s = np.dot(x, x) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * (np.dot(x[k:], x[:-k]) / n)
    return s


def nw_cov(M, lag):
    M = np.asarray(M, float); n = M.shape[0]; Mc = M - M.mean(0)
    S = Mc.T @ Mc / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        G = Mc[k:].T @ Mc[:-k] / n
        S += w * (G + G.T)
    return S


def loss(err, kind):
    err = np.asarray(err, float)
    return err ** 2 if kind == 'mse' else np.abs(err)


def clark_west(y, f_bench, f_alt, h):
    '''One-sided CW. H0: MSPE(bench) <= MSPE(alt). Large +ve => alt better.'''
    y = np.asarray(y, float); f1 = np.asarray(f_bench, float); f2 = np.asarray(f_alt, float)
    fhat = (y - f1) ** 2 - (y - f2) ** 2 + (f1 - f2) ** 2
    n = len(fhat); lag = max(h - 1, 0)
    se = np.sqrt(nw_lrv(fhat, lag) / n)
    stat = fhat.mean() / se if se > 0 else np.nan
    return stat, 1.0 - stats.norm.cdf(stat), fhat.mean()


def giacomini_white(y, f_bench, f_alt, h, kind):
    '''GW(2006). Unconditional (DM-HAC) + conditional EPA with h_t=[1,d_{t-1}].'''
    e1 = np.asarray(y, float) - np.asarray(f_bench, float)
    e2 = np.asarray(y, float) - np.asarray(f_alt, float)
    d = loss(e1, kind) - loss(e2, kind)   # +ve => alt smaller loss (better)
    n = len(d); lag = max(h - 1, 0)
    se = np.sqrt(nw_lrv(d, lag) / n)
    dm = d.mean() / se if se > 0 else np.nan
    p_un = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    hh = np.column_stack([np.ones(n - 1), d[:-1]])
    m = hh * d[1:, None]
    mbar = m.mean(0); Om = nw_cov(m, lag)
    try:
        stat = (n - 1) * mbar @ np.linalg.solve(Om, mbar)
        p_c = 1.0 - stats.chi2.cdf(stat, 2)
    except np.linalg.LinAlgError:
        stat, p_c = np.nan, np.nan
    return dm, p_un, stat, p_c, d.mean()


def sb_idx(T, B, block, rng):
    p = 1.0 / max(block, 1); idx = np.empty((B, T), dtype=np.int64)
    for b in range(B):
        cur = rng.integers(0, T)
        for t in range(T):
            if t == 0 or rng.random() < p:
                cur = rng.integers(0, T)
            else:
                cur = (cur + 1) % T
            idx[b, t] = cur
    return idx


def spa_test(D, block, B, rng):
    '''Hansen(2005) consistent SPA. D[:,k]=L(bench)-L(model_k); +mean => beats bench.'''
    n, K = D.shape; dbar = D.mean(0)
    idx = sb_idx(n, B, block, rng)
    bm = D[idx].mean(axis=1)                       # B x K
    omega = bm.std(axis=0, ddof=0) * np.sqrt(n)
    omega = np.where(omega <= 1e-12, 1e-12, omega)
    tstat = np.sqrt(n) * dbar / omega
    T_spa = max(0.0, float(tstat.max()))
    A = 0.25 * (n ** -0.25)
    thresh = -A * omega / np.sqrt(n)
    g = np.where(dbar >= thresh, dbar, 0.0)        # consistent recentering (SPA_c)
    zb = np.sqrt(n) * (bm - g) / omega
    Tb = np.maximum(0.0, zb.max(axis=1))
    return T_spa, float((Tb > T_spa).mean()), tstat, dbar, omega, int(np.argmax(tstat))


def stepm(D, names, block, B, rng, alpha=0.10):
    '''Romano-Wolf StepM: which models beat benchmark, FWER<=alpha.'''
    n, K = D.shape; dbar = D.mean(0)
    idx = sb_idx(n, B, block, rng)
    bm = D[idx].mean(axis=1)
    omega = bm.std(axis=0, ddof=0) * np.sqrt(n)
    omega = np.where(omega <= 1e-12, 1e-12, omega)
    s = np.sqrt(n) * dbar / omega
    cent = np.sqrt(n) * (bm - dbar) / omega
    active = list(range(K)); rejected = []
    while active:
        crit = np.quantile(np.max(cent[:, active], axis=1), 1 - alpha)
        newly = [k for k in active if s[k] > crit]
        if not newly:
            break
        rejected += newly; active = [k for k in active if k not in newly]
    return [names[k] for k in rejected]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='results/preds')
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5, 21, 63])
    ap.add_argument('--loss', choices=['mse', 'mae'], default='mse')
    ap.add_argument('--benchmark', default='Naive')
    ap.add_argument('--out', default='results/robustness')
    ap.add_argument('--B', type=int, default=3000)
    ap.add_argument('--blocks', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for h in a.h:
        df, y = load_preds(a.preds, h)
        names = list(df.columns)
        if a.benchmark not in names:
            print('[h=' + str(h) + '] benchmark ' + a.benchmark + ' missing; skip'); continue
        yv = y.values; fb = df[a.benchmark].values
        alts = [m for m in names if m != a.benchmark]
        rows = []
        for m in alts:
            cw_s, cw_p, cwm = clark_west(yv, fb, df[m].values, h)
            dm, p_un, gw_s, gw_p, dmean = giacomini_white(yv, fb, df[m].values, h, a.loss)
            rows.append(dict(model=m, n=len(yv), CW_stat=round(cw_s, 4), CW_p=round(cw_p, 4),
                             GW_uncond_DM=round(dm, 4), GW_uncond_p=round(p_un, 4),
                             GW_cond_chi2=round(gw_s, 4), GW_cond_p=round(gw_p, 4),
                             mean_lossdiff=dmean))
        cw_df = pd.DataFrame(rows).sort_values('CW_p')
        f1 = os.path.join(a.out, 'cw_gw_h' + str(h) + '_' + a.loss + '.csv')
        cw_df.to_csv(f1, index=False)
        e_b = yv - fb
        D = np.column_stack([loss(e_b, a.loss) - loss(yv - df[m].values, a.loss) for m in alts])
        spa_rows = []
        for blk in a.blocks:
            rng = np.random.default_rng(a.seed)
            T_spa, p, tstat, dbar, omega, best = spa_test(D, blk, a.B, rng)
            spa_rows.append(dict(block=blk, T_SPA=round(T_spa, 4), SPA_p=round(p, 4),
                                 best_model=alts[best], best_tstat=round(float(tstat[best]), 4)))
        rng = np.random.default_rng(a.seed)
        blk_mid = a.blocks[len(a.blocks) // 2] if a.blocks else 10
        winners = stepm(D, alts, blk_mid, a.B, rng)
        spa_df = pd.DataFrame(spa_rows)
        f2 = os.path.join(a.out, 'spa_h' + str(h) + '_' + a.loss + '.csv')
        spa_df.to_csv(f2, index=False)
        print('')
        print('=== h=' + str(h) + '  loss=' + a.loss + '  benchmark=' + a.benchmark + '  n=' + str(len(yv)) + ' ===')
        print('Clark-West / Giacomini-White (top 6 by CW_p; CW_p<0.05 => model beats RW):')
        print(cw_df.head(6).to_string(index=False))
        print('SPA (H0: no model beats the benchmark):')
        print(spa_df.to_string(index=False))
        print('StepM models that beat ' + a.benchmark + ' (FWER<=0.10): ' + (', '.join(winners) if winners else 'NONE'))
        print('saved -> ' + f1 + ' ; ' + f2)


if __name__ == '__main__':
    main()
