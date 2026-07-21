#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_robustness_baseline.py
=========================
MUC DICH (tra loi reviewer: "khoa cung mau 1516 co bop baseline khong?")
  Chay cac model DON GIAN (Naive, ARIMA(1,1,1), AutoARIMA) tren HAI mau:
    - COMMON  : mau chung sau dropna 24 feature (= mau chinh cua bai, ~1516 hang)
    - MAX     : mau toi da ma tung model that su can
        * Naive  chi can [Gia_target, prev_price]        -> ~1536 hang
        * ARIMA  chi can [Gia_target]                    -> ~1544 hang
  Neu MAE/RMSE/DA cua baseline ~KHONG DOI giua COMMON va MAX => viec khoa cung
  mau KHONG bop baseline; ket luan "khong danh bai duoc Naive" giu nguyen.

NGUYEN TAC (khop pipeline chinh):
  - Chi muc theo NGAY GIAO DICH (row-order sau khi bo ngay le/thieu gia) -> KHONG noi suy.
  - Walk-forward expanding: start = int(N*INITIAL_FRAC), buoc STEP.
  - Direct h-step: goc o du bao gia tai o+h. Naive: pred = gia tai o (random walk).

OUTPUT:
  results/robustness/baseline_common_vs_max.csv   (bang so sanh)
  results/preds/<model>__h<h>.csv                 (du bao tung diem tren COMMON, cho MCS)

CHAY:
  python 10_robustness_baseline.py --data data/processed/gia_cafe_master_full.csv
"""
import argparse, os
import numpy as np
import pandas as pd

# ------------------------- Cau hinh (khop _config.py) -------------------------
TARGET       = "Gia_target"
INITIAL_FRAC = 0.60
STEP         = 5
HORIZONS     = [1, 5, 21, 63]
AR     = ['target_lag1','target_lag2','target_lag3','target_ret_lag1','MA5','MA10','std5','dayofweek','month']
EXO    = ['london_vnd_kg_lag1','usdvnd_lag1','diesel','diesel_chg_1m','diesel_chg_3m','Luong_lag1m','rain_90d','oni','waterbal_90d']
SUPPLY = ['area_tn','prod_tn','yield_tn','tonkho_tan','dongia_lag1m','dongia_ret_lag1m']
F24    = AR + EXO + SUPPLY


def mae(a, p):  a=np.asarray(a,float); p=np.asarray(p,float); return float(np.mean(np.abs(a-p)))
def rmse(a, p): a=np.asarray(a,float); p=np.asarray(p,float); return float(np.sqrt(np.mean((a-p)**2)))
def da(anchor, y, p):
    anchor=np.asarray(anchor,float); y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean(np.sign(p-anchor) == np.sign(y-anchor)))


def load_master(path):
    df = pd.read_csv(path)
    for c in ['Ngay','date','Date','ngay']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]); df = df.sort_values(c).reset_index(drop=True)
            if c != 'Ngay': df = df.rename(columns={c:'Ngay'})
            break
    return df


def make_sample(df, subset):
    d = df.copy()
    d['prev_price'] = d[TARGET].shift(1)
    have = [c for c in subset if c in d.columns]
    d = d.dropna(subset=have).reset_index(drop=True)
    return d


# ------------------------- Cac ham du bao walk-forward -------------------------
def origins(N, h):
    start = int(N * INITIAL_FRAC)
    return sorted(range(N - 1 - h, start - 1, -STEP))   


def forecast_naive(price, dates, h):
    """Random-walk: du bao gia tai o+h = gia tai o."""
    rows = []
    for o in origins(len(price), h):
        rows.append((dates[o+h], price[o+h], price[o], price[o]))  # Ngay,y_true,anchor,pred
    return pd.DataFrame(rows, columns=['Ngay','y_true','anchor','pred'])


def forecast_arima(price, dates, h, order=(1,1,1), auto=False):
    """ARIMA(1,1,1) hoac AutoARIMA (neu auto=True va co pmdarima). Expanding window."""
    try:
        if auto:
            import pmdarima as pm
        else:
            from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        print(f"  [skip] thieu thu vien cho {'AutoARIMA' if auto else 'ARIMA'}: {e}")
        return None
    rows = []
    for o in origins(len(price), h):
        hist = price[:o+1]
        try:
            if auto:
                m = pm.auto_arima(hist, seasonal=False, suppress_warnings=True,
                                  error_action='ignore', stepwise=True)
                fc = float(np.asarray(m.predict(n_periods=h))[-1])
            else:
                res = ARIMA(hist, order=order,
                            enforce_stationarity=False, enforce_invertibility=False).fit()
                fc = float(np.asarray(res.forecast(steps=h))[-1])
        except Exception:
            fc = float(price[o])  # fallback naive neu fit loi
        rows.append((dates[o+h], price[o+h], price[o], fc))
    return pd.DataFrame(rows, columns=['Ngay','y_true','anchor','pred'])


def eval_df(pred_df):
    return dict(n=len(pred_df),
                MAE=round(mae(pred_df.y_true, pred_df.pred),1),
                RMSE=round(rmse(pred_df.y_true, pred_df.pred),1),
                DA=round(da(pred_df.anchor, pred_df.y_true, pred_df.pred),4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/gia_cafe_master_full.csv')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--auto', action='store_true', help='them AutoARIMA (can pmdarima)')
    a = ap.parse_args()

    df = load_master(a.data)
    common   = make_sample(df, [TARGET,'prev_price']+F24)     # ~1516
    naive_mx = make_sample(df, [TARGET,'prev_price'])          # ~1536
    arima_mx = make_sample(df, [TARGET])                       # ~1544
    print(f"[mau] COMMON={len(common)} | NAIVE_MAX={len(naive_mx)} | ARIMA_MAX={len(arima_mx)}")

    preds_dir = os.path.join(a.outdir, 'preds'); os.makedirs(preds_dir, exist_ok=True)
    rob_dir   = os.path.join(a.outdir, 'robustness'); os.makedirs(rob_dir, exist_ok=True)

    table = []
    for h in HORIZONS:
        # ---- Naive: COMMON vs MAX ----
        pc = forecast_naive(common[TARGET].values, common['Ngay'].values, h)
        pm_ = forecast_naive(naive_mx[TARGET].values, naive_mx['Ngay'].values, h)
        ec, em = eval_df(pc), eval_df(pm_)
        table.append(dict(model='Naive', h=h, sample='COMMON', **ec))
        table.append(dict(model='Naive', h=h, sample='MAX',    **em))
        pc.to_csv(os.path.join(preds_dir, f'Naive__h{h}.csv'), index=False)

        # ---- ARIMA(1,1,1): COMMON vs MAX ----
        ac = forecast_arima(common[TARGET].values, common['Ngay'].values, h)
        am = forecast_arima(arima_mx[TARGET].values, arima_mx['Ngay'].values, h)
        if ac is not None:
            table.append(dict(model='ARIMA(1,1,1)', h=h, sample='COMMON', **eval_df(ac)))
            ac.to_csv(os.path.join(preds_dir, f'ARIMA__h{h}.csv'), index=False)
        if am is not None:
            table.append(dict(model='ARIMA(1,1,1)', h=h, sample='MAX', **eval_df(am)))

        # ---- AutoARIMA (tuy chon) ----
        if a.auto:
            qc = forecast_arima(common[TARGET].values, common['Ngay'].values, h, auto=True)
            qm = forecast_arima(arima_mx[TARGET].values, arima_mx['Ngay'].values, h, auto=True)
            if qc is not None:
                table.append(dict(model='AutoARIMA', h=h, sample='COMMON', **eval_df(qc)))
                qc.to_csv(os.path.join(preds_dir, f'AutoARIMA__h{h}.csv'), index=False)
            if qm is not None:
                table.append(dict(model='AutoARIMA', h=h, sample='MAX', **eval_df(qm)))
        print(f"[h={h}] xong")

    out = pd.DataFrame(table).sort_values(['model','h','sample']).reset_index(drop=True)
    fp = os.path.join(rob_dir, 'baseline_common_vs_max.csv')
    out.to_csv(fp, index=False)
    print('\n=== BANG ROBUSTNESS (COMMON vs MAX) ===')
    print(out.to_string(index=False))
    print(f'\n-> luu: {fp}')
    print(f'-> du bao tung diem (COMMON) da luu vao: {preds_dir}/  (dung cho 12_mcs.py)')


if __name__ == '__main__':
    main()