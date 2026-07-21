#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_lstm_gru.py  --  Bo sung LSTM & GRU vao benchmark, DONG NHAT giao thuc voi
run_dl_models.py (NHITS/NBEATSx/Chronos).

Muc dich: chinh ban chay mot LSTM/GRU TRUNG THUC (cung expanding walk-forward,
cung baseline Naive can khop tung diem) de kiem chung truc tiep cac cong bo
'LSTM chinh xac >98%' -- ky vong ket qua se THUA random walk tren muc gia,
cung co ket luan H1.

Diem giong run_dl_models.py:
  - load_data(): can khop 18 feature -> N=1516, Naive MAE trung khit
    1567 / 3511 / 8646 / 15946 (h=1/5/21/63).
  - UNIVARIATE (chi dung gia) -> apples-to-apples voi NHITS/NBEATSx/Chronos.
  - ds = chi so nguyen, freq=1 (tranh lech lich ngay nghi/le).
  - n_windows = len(range(start, N-h, STEP)); Naive = gia tai cutoff.
  - dm_test() Newey-West giong het.
  - --refit_every 1 => true expanding (refit moi origin). MAC DINH = 1.
Diem khac:
  - LSTM/GRU la RNN nen KHONG bo qua h=1 (chay ca 4 chan troi).

Vi du:
  python run_lstm_gru.py                 # true expanding, ca 4 chan troi
  python run_lstm_gru.py --horizons 5 21 63
  python run_lstm_gru.py --refit_every 0 # fit 1 lan (nhanh, de doi chieu)
"""
from openpyxl.workbook import defined_name
import argparse, warnings, time, os
import numpy as np
import pandas as pd
from math import erf, sqrt
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, r"E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\scripts\\model")  # noi de export_preds.py
from export_preds import save_preds
PREDS_DIR = r"E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\preds"

DATA_PATH    = "data/processed/gia_cafe_master_full.csv"
TARGET       = "Gia_target"
DATE_COL     = "date"
HORIZONS     = [1, 5, 21, 63]
INITIAL_FRAC = 0.6
STEP         = 5
OUT_DIR      = "E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\"

def _now(): return time.strftime("%H:%M:%S")

def load_data():
    global DATE_COL
    df = pd.read_csv(DATA_PATH)
    if DATE_COL not in df.columns:
        for c in ["date","Date","Ngay","ngay"]:
            if c in df.columns: DATE_COL = c; break
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["prev_price"] = df[TARGET].shift(1)
    before = len(df)
    _ALIGN_FEATURES = ['target_lag1','target_lag2','target_lag3','target_ret_lag1',
        'MA5','MA10','std5','dayofweek','month','london_vnd_kg_lag1','usdvnd_lag1',
        'diesel','diesel_chg_1m','diesel_chg_3m','Luong_lag1m','rain_90d','oni','waterbal_90d']
    _have = [c for c in _ALIGN_FEATURES if c in df.columns]
    df = df.dropna(subset=[TARGET,"prev_price"]+_have).reset_index(drop=True)
    print(f"[load] giu {len(df)}/{before} hang (N={len(df)}, start={int(len(df)*INITIAL_FRAC)}) [CAN KHOP 1516]", flush=True)
    return df

def dm_test(em, en, h=1):
    e1, e2 = np.asarray(em,float), np.asarray(en,float)
    d = np.abs(e1) - np.abs(e2); n = len(d)
    if n == 0: return float("nan"), float("nan")
    dbar = d.mean(); var = np.var(d, ddof=0)
    for k in range(1, h):
        if n-k > 1: var += 2*(1-k/h)*np.cov(d[k:], d[:-k])[0,1]
    if var <= 0: return float("nan"), float("nan")
    dm = dbar / sqrt(var/n)
    p  = 2*(1 - 0.5*(1 + erf(abs(dm)/sqrt(2))))
    return float(dm), float(p)

def run_lstm_gru(df, horizons, rows, refit_every=1):
    print("\n" + "="*60)
    print("LSTM / GRU  (UNIVARIATE, can khop walk-forward)")
    print("="*60)
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import LSTM, GRU
    except ImportError:
        print("[!] pip install neuralforecast"); return
    import torch as _torch

    N     = len(df)
    start = int(N * INITIAL_FRAC)
    serie = df[TARGET].astype(float).values
    long  = pd.DataFrame({"unique_id": "cafe", "ds": np.arange(N), "y": serie})

    _use_gpu = _torch.cuda.is_available()
    _accel   = "gpu" if _use_gpu else "cpu"
    _devices = 1 if _use_gpu else None
    if _use_gpu: _torch.set_float32_matmul_precision("high")
    print(f"[device] CUDA={_use_gpu} -> accelerator='{_accel}'", flush=True)

    _d2v = {int(i): float(serie[int(i)]) for i in range(N)}
    model_specs = [("LSTM", LSTM), ("GRU", GRU)]

    for h in horizons:
        n_windows = len(range(start, N - h, STEP))
        print(f"\n[{_now()}] LSTM/GRU h={h}  n_windows={n_windows} "
              f"(cutoff som nhat ~ index {N - h - (n_windows-1)*STEP}, start={start})", flush=True)
        for tag, Cls in model_specs:
            t0 = time.time()
            try:
                mdl = Cls(h=h, input_size=max(2*h,30), max_steps=500,
                          encoder_hidden_size=128, scaler_type="robust",
                          accelerator=_accel, devices=_devices,
                          enable_progress_bar=False, enable_model_summary=False)
                nf = NeuralForecast(models=[mdl], freq=1)
                if refit_every and refit_every > 0:
                    try:
                        cv = nf.cross_validation(df=long, n_windows=n_windows, step_size=STEP,
                                                 refit=refit_every, verbose=False)
                    except Exception as _e:
                        print(f"  [warn] {tag} refit loi ({type(_e).__name__}); fit 1 lan.", flush=True)
                        cv = nf.cross_validation(df=long, n_windows=n_windows, step_size=STEP, verbose=False)
                else:
                    cv = nf.cross_validation(df=long, n_windows=n_windows, step_size=STEP, verbose=False)

                cv_h = (cv.sort_values("ds").groupby("cutoff", sort=False).tail(1).reset_index(drop=True))
                if tag not in cv_h.columns:
                    print(f"  [!] khong thay cot {tag} trong ket qua CV", flush=True); continue
                m  = cv_h[tag].notna() & cv_h["y"].notna()
                yt = cv_h.loc[m,"y"].values.astype(float)
                yp = cv_h.loc[m,tag].values.astype(float)
                yn = cv_h.loc[m,"cutoff"].map(_d2v).values.astype(float)
                ok = ~np.isnan(yn); yt,yp,yn = yt[ok],yp[ok],yn[ok]
                cut = cv_h.loc[m,"cutoff"].values.astype(int)[ok]   # cutoff = chi so goc (o)
                dts = pd.to_datetime(df[DATE_COL].values[cut + h])   # diem du bao = o+h
                save_preds(tag, h, dts, yt, yp, outdir=PREDS_DIR)    # tag = "LSTM" hoac "GRU"
                mm=float(np.mean(np.abs(yt-yp))); nm=float(np.mean(np.abs(yt-yn)))
                rm=float(np.sqrt(np.mean((yt-yp)**2)))
                da=float(np.mean(np.sign(yt-yn)==np.sign(yp-yn)))
                dm,p=dm_test(yt-yp, yt-yn, h=h)
                rows.append(dict(model=tag,h=h,n=int(len(yt)),MAE=round(mm,1),
                                 MAE_naive=round(nm,1),
                                 dMAE_pct=round(100*(mm/nm-1),2) if nm>0 else None,
                                 RMSE=round(rm,1),DA=round(da,3),DM=round(dm,3),DM_p=round(p,4)))
                print(f"  {tag:6s} h={h:2d}: MAE={mm:9.1f}  Naive={nm:9.1f}  "
                      f"dMAE%={100*(mm/nm-1):+6.2f}  DA={da:.3f}  DM_p={p:.4f}  n={len(yt)}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  => LOI {tag} h={h}: {type(e).__name__}: {e}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--horizons",type=int,nargs="+",default=HORIZONS)
    ap.add_argument("--refit_every",type=int,default=1,
                    help="1=true expanding refit moi origin (mac dinh); 0=fit 1 lan (nhanh)")
    a=ap.parse_args()
    print(f"[{_now()}] Load..."); df=load_data()
    rows=[]
    run_lstm_gru(df, a.horizons, rows, a.refit_every)
    out=pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_DIR+"lstm_gru_comparison.csv", index=False)
    print(f"\n[done] -> {OUT_DIR}lstm_gru_comparison.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
