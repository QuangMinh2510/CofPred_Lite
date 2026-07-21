#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
grid_experiment.py -- quantify the grid-misalignment artifact (referee 3.4a)
============================================================================
Reproduce, on OUR OWN models, the old MISALIGNED grid comparison vs the UNIFIED
last-anchored grid, to measure whether deep learning looks spuriously better
under misalignment and loses under the unified grid.

Grids for horizon h (n = aligned sample length, start = int(n*0.60)):
  UNIFIED  O_U = sorted(range(n-1-h, start-1, -STEP))   # last-anchored (current paper)
  FORWARD  O_F = list(range(start, n-h, STEP))          # start-anchored (the OLD stat/ML grid)

Minimal-retraining design (keeps canonical numbers intact, no neural drift):
  * Statistical/ML panel (Naive, ARIMA(1,1,1), Ridge on F18) is re-fit on BOTH
    grids -- cheap, deterministic, statsmodels/scikit-learn only.
  * Deep-learning models REUSE the existing saved forecasts in results/preds
    (already produced on the end-anchored == unified grid). Nothing is retrained.
  * MISALIGNED comparison: stat/ML scored on O_F, DL scored on O_U (different dates).
    UNIFIED    comparison: every model scored on O_U (identical dates).

We report MAE (VND/kg) and sMAPE(%). sMAPE is scale-free, so it is NOT confounded
by the two grids covering periods with different price levels -- read the sMAPE
columns for the cleanest read of the artifact. The script prints an explicit,
data-driven verdict for each horizon (it does not assume any particular outcome).

Run (cofpred env; needs statsmodels + scikit-learn):
  python grid_experiment.py --h 5 21 63 --dl LSTM NHITS GRU NBEATSx --out results/grid_experiment
'''
import argparse, os, sys
import numpy as np
import pandas as pd

# ---- reuse the project's exact alignment if _config.py is importable ----
TARGET = 'Gia_target'; INITIAL_FRAC = 0.60; STEP = 5
_AR = ['target_lag1', 'target_lag2', 'target_lag3', 'target_ret_lag1',
       'MA5', 'MA10', 'std5', 'dayofweek', 'month']
_EXO = ['london_vnd_kg_lag1', 'usdvnd_lag1', 'diesel', 'diesel_chg_1m',
        'diesel_chg_3m', 'Luong_lag1m', 'rain_90d', 'oni', 'waterbal_90d']
F18 = _AR + _EXO

_cfg = None
_here = os.path.dirname(os.path.abspath(__file__))
for _c in [os.path.join(_here, 'analysis'), os.path.join(_here, '..', 'analysis'),
           os.path.join(_here, 'scripts', 'analysis'), _here]:
    if os.path.exists(os.path.join(_c, '_config.py')):
        sys.path.insert(0, _c)
        try:
            import _config as _cfg
            TARGET, INITIAL_FRAC, STEP, F18 = _cfg.TARGET, _cfg.INITIAL_FRAC, _cfg.STEP, _cfg.F18
        except Exception:
            _cfg = None
        break


def load_master(path):
    if _cfg is not None:
        return _cfg.load_master(path)
    df = pd.read_csv(path)
    for c in ['date', 'Date', 'Ngay', 'ngay']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]); df = df.sort_values(c).reset_index(drop=True)
            if c != 'Ngay':
                df = df.rename(columns={c: 'Ngay'})
            break
    return df


def align_frame(df, feats):
    if _cfg is not None:
        return _cfg.align_frame(df, feats)
    d = df.copy(); d['prev_price'] = d[TARGET].shift(1)
    have = [c for c in feats if c in d.columns]
    d = d.dropna(subset=[TARGET, 'prev_price'] + have).reset_index(drop=True)
    print('[align] rows=' + str(len(d)) + ' features=' + str(len(have)))
    return d, have


def mae(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    return float(np.mean(np.abs(a - p)))


def smape(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    den = np.abs(a) + np.abs(p)
    return float(np.mean(np.where(den > 0, 2.0 * np.abs(a - p) / den, 0.0)) * 100.0)


def O_unified(n, h, start):
    return sorted(range(n - 1 - h, start - 1, -STEP))


def O_forward(n, h, start):
    return list(range(start, n - h, STEP))


def run_naive(price, origins, h):
    yt, pr = [], []
    for d in origins:
        if d + h >= len(price):
            continue
        yt.append(price[d + h]); pr.append(price[d])
    return np.array(yt), np.array(pr)


def run_arima(price, origins, h, order=(1, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    yt, pr = [], []
    for d in origins:
        if d + h >= len(price):
            continue
        try:
            fc = ARIMA(price[:d + 1], order=order).fit().forecast(h)[-1]
        except Exception:
            fc = price[d]
        yt.append(price[d + h]); pr.append(float(fc))
    return np.array(yt), np.array(pr)


def run_ridge(X, price, origins, h, alpha=1.0, min_train=60):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    yt, pr = [], []
    for d in origins:
        if d + h >= len(price):
            continue
        tr = d - h + 1
        if tr < min_train:
            continue
        Xtr = X[:tr]; ytr = price[h:h + tr] - price[:tr]      # delta target
        sc = StandardScaler().fit(Xtr)
        mdl = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
        raw = float(mdl.predict(sc.transform(X[d:d + 1]))[0])
        yt.append(price[d + h]); pr.append(price[d] + raw)
    return np.array(yt), np.array(pr)


def load_dl(preds_dir, model, h):
    f = os.path.join(preds_dir, model + '__h' + str(h) + '.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return np.asarray(d['y_true'], float), np.asarray(d['pred'], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/gia_cafe_master_full.csv')
    ap.add_argument('--preds', default='results/preds')
    ap.add_argument('--h', type=int, nargs='+', default=[5, 21, 63])
    ap.add_argument('--dl', nargs='+', default=['LSTM', 'NHITS', 'GRU', 'NBEATSx'])
    ap.add_argument('--out', default='results/grid_experiment')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_master(a.data)
    df, have = align_frame(df, F18)
    price = df[TARGET].values.astype(float)
    X = df[[c for c in F18 if c in have]].values.astype(float)
    n = len(df); start = int(n * INITIAL_FRAC)
    print('n=' + str(n) + '  start=' + str(start) + '  STEP=' + str(STEP))

    for h in a.h:
        OU = O_unified(n, h, start); OF = O_forward(n, h, start)
        print(''); print('===== horizon h=' + str(h) + ' =====')
        print('|O_U(unified)|=' + str(len(OU)) + '  |O_F(forward)|=' + str(len(OF)))

        rows = []
        # ---- statistical/ML on BOTH grids ----
        for label, fn in [('Naive', lambda o: run_naive(price, o, h)),
                          ('ARIMA', lambda o: run_arima(price, o, h)),
                          ('Ridge', lambda o: run_ridge(X, price, o, h))]:
            yU, pU = fn(OU); yF, pF = fn(OF)
            rows.append(dict(model=label, group='stat/ML',
                             MAE_unified=mae(yU, pU), sMAPE_unified=smape(yU, pU),
                             MAE_forward=mae(yF, pF), sMAPE_forward=smape(yF, pF)))
        # sanity: our Naive on O_U must match the saved Naive preds
        sav = load_dl(a.preds, 'Naive', h)
        if sav is not None:
            yU, pU = run_naive(price, OU, h)
            print('[sanity] Naive MAE ours(O_U)=' + str(round(mae(yU, pU), 3)) +
                  '  saved=' + str(round(mae(sav[0], sav[1]), 3)))
        # ---- deep learning: reuse saved (end-anchored == unified) preds ----
        for m in a.dl:
            dl = load_dl(a.preds, m, h)
            if dl is None:
                print('[skip] no saved preds for ' + m + ' at h=' + str(h)); continue
            rows.append(dict(model=m, group='DL',
                             MAE_unified=mae(dl[0], dl[1]), sMAPE_unified=smape(dl[0], dl[1]),
                             MAE_forward=np.nan, sMAPE_forward=np.nan))

        res = pd.DataFrame(rows)
        # misaligned metric: stat/ML from FORWARD grid, DL from UNIFIED grid
        res['MAE_misaligned'] = np.where(res['group'] == 'stat/ML', res['MAE_forward'], res['MAE_unified'])
        res['sMAPE_misaligned'] = np.where(res['group'] == 'stat/ML', res['sMAPE_forward'], res['sMAPE_unified'])
        res['rank_unified'] = res['MAE_unified'].rank(method='min').astype(int)
        res['rank_misaligned'] = res['MAE_misaligned'].rank(method='min').astype(int)
        res = res.sort_values('rank_unified').reset_index(drop=True)
        fout = os.path.join(a.out, 'grid_experiment_h' + str(h) + '.csv')
        res.to_csv(fout, index=False)

        show = res[['model', 'group', 'MAE_unified', 'MAE_misaligned',
                    'sMAPE_unified', 'sMAPE_misaligned', 'rank_unified', 'rank_misaligned']]
        print(show.round(3).to_string(index=False))

        # data-driven verdict
        naive_u = res.loc[res['model'] == 'Naive', 'rank_unified'].iloc[0]
        naive_m = res.loc[res['model'] == 'Naive', 'rank_misaligned'].iloc[0]
        dl_rows = res[res['group'] == 'DL']
        flipped = dl_rows[(dl_rows['rank_misaligned'] < naive_m) & (dl_rows['rank_unified'] > naive_u)]
        print('Naive rank: unified=' + str(naive_u) + '  misaligned=' + str(naive_m))
        if len(flipped):
            print('VERDICT: grid misalignment creates a SPURIOUS DL win for: ' +
                  ', '.join(flipped['model'].tolist()) +
                  ' (rank above Naive when misaligned, below when unified).')
        else:
            print('VERDICT: no DL model overtakes Naive purely from misalignment at this horizon '
                  '(report the sMAPE gap between the unified and misaligned columns).')
        print('saved -> ' + fout)


if __name__ == '__main__':
    main()
