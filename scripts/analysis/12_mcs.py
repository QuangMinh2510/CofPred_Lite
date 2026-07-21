#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_mcs.py  --  Model Confidence Set (Hansen, Lunde & Nason 2011)
=================================================================
MUC DICH: thay cho hang loat DM tung cap, chay MCS de tim TAP MODEL TOT NHAT
  (Superior Set of Models) o muc tin cay cho truoc -> khien reviewer het duong
  van "so sanh boi khi test nhieu model".

DAU VAO: thu muc results/preds/ chua cac file du bao tung diem CUNG mau (COMMON),
  moi file 1 model cho 1 horizon, ten dang <MODEL>__h<h>.csv, cot: Ngay,y_true,pred
  (anchor la tuy chon). Tao boi 10_robustness_baseline.py va export_preds.py.

THONG KE: dung "range statistic" T_R voi STATIONARY BLOCK BOOTSTRAP
  (chi phu thuoc numpy/pandas, khong can thu vien ngoai).

CHAY:
  python 12_mcs.py --preds results/preds --h 5 --loss mae --alpha 0.10
  python 12_mcs.py --preds results/preds --h 5 --loss mse --block 10 --B 3000
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd


# ============================================================
# BO SUNG: bang FULL-METRIC (MAE, RMSE, sMAPE, MASE, DA) cho MOI model.
# Doc tu results/preds/ (du bao DA LUU) -> KHONG train lai model,
# nen so lieu TRUNG KHIT voi bang MCS & bang ML hien co.
# ============================================================
def _mae(a, p):  return float(np.mean(np.abs(a - p)))
def _rmse(a, p): return float(np.sqrt(np.mean((a - p) ** 2)))

def _smape(a, p):
    denom = (np.abs(a) + np.abs(p)) / 2.0
    return float(np.mean(np.abs(a - p) / np.where(denom == 0, np.nan, denom)) * 100)

def _mase(a, p, d):
    # MASE = MAE(model) / d ; d = MAE naive 1-buoc tren PHAN TRAIN BAN DAU (60%)
    return float(_mae(a, p) / d)

def _da(a, p, prev):
    m = np.isfinite(prev)
    if int(m.sum()) == 0:
        return float('nan')
    return float(np.mean(np.sign(a[m] - prev[m]) == np.sign(p[m] - prev[m])) * 100)

def get_mase_denom(ml_csv):
    """Lay mau so MASE d = MAE/MASE tu 1 dong bang ML (uu tien Naive)
    -> dam bao MASE cua MOI model dung CUNG thang do voi nhom ML."""
    m = pd.read_csv(ml_csv)
    r = m[m['Model'] == 'Naive']
    r = r.iloc[0] if len(r) else m.iloc[0]
    return float(r['MAE']) / float(r['MASE'])

def full_metrics(preds_dir, h, d):
    """Tinh MAE, RMSE, sMAPE, MASE, DA cho MOI model tai horizon h,
    tren CUNG mau ngay giao nhau (giong MCS)."""
    files = sorted(glob.glob(os.path.join(preds_dir, f'*__h{h}.csv')))
    if not files:
        raise SystemExit(f'Khong tim thay *__h{h}.csv trong {preds_dir}')
    raw = {}
    for f in files:
        name = re.sub(r'__h\d+\.csv$', '', os.path.basename(f))
        dfp = pd.read_csv(f)
        dfp['Ngay'] = pd.to_datetime(dfp['Ngay'])
        raw[name] = dfp.set_index('Ngay').sort_index()
    common = None
    for dfp in raw.values():
        idx = dfp.dropna(subset=['y_true', 'pred']).index
        common = idx if common is None else common.intersection(idx)
    rows = []
    for name, dfp in raw.items():
        b = dfp.reindex(common)
        a = b['y_true'].astype(float).values
        p = b['pred'].astype(float).values
        prev = (b['anchor'].astype(float).values
                if 'anchor' in b.columns else np.full(len(a), np.nan))
        rows.append({
            'model':    name,
            'MAE':      round(_mae(a, p), 1),
            'RMSE':     round(_rmse(a, p), 1),
            'sMAPE(%)': round(_smape(a, p), 3),
            'MASE':     round(_mase(a, p, d), 3),
            'DA(%)':    round(_da(a, p, prev), 1),
        })
    return pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)


def load_losses(preds_dir, h, loss='mae'):
    """Doc moi file <MODEL>__h<h>.csv, can chinh theo Ngay giao nhau, tra ve
    (models, dates, L) voi L shape (T, M) la loss tung diem."""
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
    """Politis-Romano stationary bootstrap: tra ve mang chi so (B, T)."""
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


def mcs(L, alpha=0.10, B=3000, block=10, seed=1234):
    """MCS range-statistic. L: (T, M) loss. Tra ve list dict xep hang loai + p_MCS."""
    T, M = L.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(T, B, block, rng)
    # bootstrap mean loss cho tung model: bmean[b, i]
    bmean = L[idx].mean(axis=1)          # (B, M)
    full  = L.mean(axis=0)               # (M,)
    z = bmean - full                     # (B, M): sai lech bootstrap quanh trung binh

    included = list(range(M))
    elim_order, elim_p = [], []
    running = 0.0
    while len(included) > 1:
        sub = np.array(included); k = len(sub)
        Lm = full[sub]                   # (k,)
        zk = z[:, sub]                   # (B, k)
        # phuong sai bootstrap cua chenh lech cap d_ij = Lm_i - Lm_j
        var_ij = np.zeros((k, k))
        for i in range(k):
            diff = zk[:, i][:, None] - zk      # (B, k)
            var_ij[i] = diff.var(axis=0)
        dij = Lm[:, None] - Lm[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            tij = np.where(var_ij > 0, dij / np.sqrt(var_ij), 0.0)
        TR = np.nanmax(np.abs(tij))
        # phan phoi bootstrap cua T_R duoi null
        TR_b = np.empty(B)
        for b in range(B):
            zb = zk[b]
            db = zb[:, None] - zb[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                tb = np.where(var_ij > 0, db / np.sqrt(var_ij), 0.0)
            TR_b[b] = np.nanmax(np.abs(tb))
        p = float(np.mean(TR_b >= TR))
        running = max(running, p)         # p_MCS = luy tich max
        if p >= alpha:
            break                         # tat ca model con lai deu nam trong MCS
        # loai model "te nhat": t_i = (Lm_i - mean_k Lm) / sd_bootstrap
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
    ap.add_argument('--ml_csv', default='results/ML_model/model_comparison_h1_level.csv',
                    help='bang ML de lay mau so MASE (d) cho nhat quan thang do')
    a = ap.parse_args()

    models, dates, L = load_losses(a.preds, a.h, a.loss)
    print(f'[MCS] h={a.h} loss={a.loss} | {len(models)} model | T={L.shape[0]} diem chung '
          f'({pd.Timestamp(dates.min()).date()} -> {pd.Timestamp(dates.max()).date()})')
    rows = mcs(L, alpha=a.alpha, B=a.B, block=a.block)
    df = pd.DataFrame(rows)
    df['model'] = df['model_idx'].map(lambda i: models[i])
    df['MAE_or_MSE'] = df['model_idx'].map(lambda i: round(float(L[:, i].mean()), 2))
    df = df[['model','status','p_MCS','MAE_or_MSE']].sort_values(
        ['status','MAE_or_MSE'], ascending=[True, True]).reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, f'mcs_h{a.h}_{a.loss}.csv')
    df.to_csv(fp, index=False)
    print('\n=== KET QUA MCS (alpha={:.2f}) ==='.format(a.alpha))
    print(df.to_string(index=False))
    print(f'\nModel nam trong MCS (khong the loai o muc tin cay {1-a.alpha:.0%}):',
          ', '.join(df.loc[df.status=='IN_MCS','model']))
    print(f'-> luu: {fp}')

    # --- BO SUNG: bang FULL-METRIC (chi phu thuoc horizon, khong phu thuoc --loss) ---
    try:
        d = get_mase_denom(a.ml_csv)
        fm = full_metrics(a.preds, a.h, d)
        os.makedirs(a.out, exist_ok=True)
        fmfp = os.path.join(a.out, f'full_metrics_h{a.h}.csv')
        fm.to_csv(fmfp, index=False)
        print(f'\n=== FULL METRICS (h={a.h}) | mau so MASE d={d:.2f} ===')
        print(fm.to_string(index=False))
        chk = fm[fm['model'] == 'Naive']
        if len(chk):
            print(f"[kiem tra] Naive: sMAPE={chk['sMAPE(%)'].iloc[0]} MASE={chk['MASE'].iloc[0]} "
                  f"(ky vong ~1.459 & ~5.228 tai h=1)")
        print(f'-> luu: {fmfp}')
    except Exception as e:
        print(f'[full_metrics] bo qua ({e})')


if __name__ == '__main__':
    main()