#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dl_models_aligned.py  (v2)  --  ban DA CAN KHOP + DA SUA loi futr_df.

Thay doi so voi v1:
  - NHITS/NBEATSx chuyen sang UNIVARIATE (bo futr_exog_list).
    Ly do: futr_exog doi gia tri exog TUONG LAI (khong co) -> loi
    'missing combinations of ids and times in futr_df'. Bo exog vua tranh loi,
    vua dong nhat voi Chronos (cung univariate) -> so sanh apples-to-apples.
  - Dung CHI SO NGUYEN cho ds (freq=1) thay cho ngay lam viec 'B'
    -> tranh lech lich do ngay nghi/le.
  - n_windows = len(range(start, N-h, STEP)) -> cutoff som nhat ~ int(0.6N),
    cung span test voi statistical models; Naive = gia tai cutoff
    -> MAE_naive trung 3511.4 / 8646.5 / 15946.0.

Chronos giu NGUYEN (da chay dung & da can khop).
Vi du: python run_dl_models_aligned.py --horizons 5 21 63
"""
import argparse, warnings, time
import numpy as np
import pandas as pd
import os
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
    # CAN KHOP MAU voi statistical/ML pipeline (18 feature AR+EXO) -> N=1516,
    # Naive MAE trung khit 3511.4 / 8646.5 / 15946.0 (h5/h21/h63).
    # DL VAN UNIVARIATE (chi dung gia); dropna theo feature chi de KHOA cung
    # mau danh gia -> 1 cot Naive chung cho ca bang benchmark.
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

# ============================================================
# A) NHITS + NBEATSx  -- UNIVARIATE, CAN KHOP
# ============================================================
def run_nhits_nbeatsx(df, horizons, rows, refit_every=0):
    print("\n" + "="*60)
    print("A) NHITS / NBEATSx  (UNIVARIATE, can khop walk-forward)")
    print("="*60)
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, NBEATSx
    except ImportError:
        print("[!] pip install neuralforecast"); return
    import torch as _torch

    N     = len(df)
    start = int(N * INITIAL_FRAC)
    serie = df[TARGET].astype(float).values
    # ds = CHI SO NGUYEN (tranh loi lich ngay nghi); univariate (chi y)
    long = pd.DataFrame({"unique_id": "cafe", "ds": np.arange(N), "y": serie})

    _use_gpu = _torch.cuda.is_available()
    _accel   = "gpu" if _use_gpu else "cpu"
    _devices = 1 if _use_gpu else None
    if _use_gpu: _torch.set_float32_matmul_precision("high")
    print(f"[device] CUDA={_use_gpu} -> accelerator='{_accel}'", flush=True)

    for h in horizons:
        t0 = time.time()
        if h == 1:
            print("  [skip] NHITS/NBEATSx khong ho tro h=1", flush=True); continue
        n_windows = len(range(start, N - h, STEP))
        print(f"\n[{_now()}] NHITS+NBEATSx h={h}  n_windows={n_windows} "
              f"(cutoff som nhat ~ index {N - h - (n_windows-1)*STEP}, start={start})", flush=True)
        try:
            models = [
                NHITS(h=h, input_size=max(2*h,30), max_steps=500, n_blocks=[1,1,1],
                      scaler_type="robust", accelerator=_accel, devices=_devices,
                      enable_progress_bar=False, enable_model_summary=False),
                NBEATSx(h=h, input_size=max(2*h,30), max_steps=500,
                        scaler_type="robust", accelerator=_accel, devices=_devices,
                        enable_progress_bar=False, enable_model_summary=False),
            ]
            nf = NeuralForecast(models=models, freq=1)   # freq nguyen
            if refit_every and refit_every > 0:
                try:
                    cv = nf.cross_validation(df=long, n_windows=n_windows, step_size=STEP,
                                             refit=refit_every, verbose=False)
                except Exception as _e:
                    print(f"  [warn] refit loi ({type(_e).__name__}); fit 1 lan.", flush=True)
                    cv = nf.cross_validation(df=long, n_windows=n_windows, step_size=STEP, verbose=False)
            else:
                cv = nf.cross_validation(df=long, n_windows=n_windows, step_size=STEP, verbose=False)

            cv_h = (cv.sort_values("ds").groupby("cutoff", sort=False).tail(1).reset_index(drop=True))
            _d2v = {int(i): float(serie[int(i)]) for i in range(N)}
            for tag in ["NHITS", "NBEATSx"]:
                if tag not in cv_h.columns: continue
                m  = cv_h[tag].notna() & cv_h["y"].notna()
                yt = cv_h.loc[m,"y"].values.astype(float)
                yp = cv_h.loc[m,tag].values.astype(float)
                yn = cv_h.loc[m,"cutoff"].map(_d2v).values.astype(float)
                ok = ~np.isnan(yn); yt,yp,yn = yt[ok],yp[ok],yn[ok]
                cut = cv_h.loc[m,"cutoff"].values.astype(int)[ok]   # cutoff = chi so goc (o)
                dts = pd.to_datetime(df[DATE_COL].values[cut + h])  # diem du bao = o+h
                save_preds(tag, h, dts, yt, yp, outdir=PREDS_DIR)
                mm=float(np.mean(np.abs(yt-yp))); nm=float(np.mean(np.abs(yt-yn)))
                rm=float(np.sqrt(np.mean((yt-yp)**2)))
                da=float(np.mean(np.sign(yt-yn)==np.sign(yp-yn)))
                dm,p=dm_test(yt-yp, yt-yn, h=h)
                rows.append(dict(model=tag,h=h,n=int(len(yt)),MAE=round(mm,1),
                                 MAE_naive=round(nm,1),
                                 dMAE_pct=round(100*(mm/nm-1),2) if nm>0 else None,
                                 RMSE=round(rm,1),DA=round(da,3),DM=round(dm,3),DM_p=round(p,4)))
                print(f"  {tag:8s} h={h:2d}: MAE={mm:9.1f}  Naive={nm:9.1f}  "
                      f"dMAE%={100*(mm/nm-1):+6.2f}  DA={da:.3f}  DM_p={p:.4f}  n={len(yt)}", flush=True)
            print(f"  => OK {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  => LOI h={h}: {type(e).__name__}: {e}", flush=True)

# ============================================================
# B) Chronos zero-shot  (GIU NGUYEN - da chay dung)
# ============================================================
def run_chronos_all(serie, horizons, rows, context=512, dates=None):
    print("\n" + "="*60); print("B) Chronos zero-shot (can khop)"); print("="*60)
    try:
        import torch; from chronos import ChronosPipeline
    except ImportError:
        print("[!] pip install chronos-forecasting torch"); return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe=None
    for mid,t,kw in [("amazon/chronos-bolt-base","bolt-base",{"dtype":torch.float32}),
                     ("amazon/chronos-t5-small","t5-small",{"dtype":torch.bfloat16})]:
        try:
            pipe=ChronosPipeline.from_pretrained(mid, device_map=_device, **kw)
            print(f"[{_now()}] Chronos '{t}' on {_device.upper()}", flush=True); break
        except Exception as e:
            print(f"[!] '{t}' fail ({type(e).__name__}); fallback...", flush=True)
    if pipe is None: print("[!] Khong load duoc Chronos."); return
    n=len(serie); start=int(n*INITIAL_FRAC)
    for h in horizons:
        t0=time.time(); yt=[];yp=[];yn=[];base=[]; origins=[]
        try:
            for d in sorted(range(n - 1 - h, start - 1, -STEP)):
                ctx=torch.tensor(serie[max(0,d-context):d+1],dtype=torch.float32)
                fc=pipe.predict(ctx,prediction_length=h)
                yp.append(float(np.quantile(fc[0].numpy(),0.5,axis=0)[-1]))
                yt.append(float(serie[d+h])); yn.append(float(serie[d])); base.append(float(serie[d])); origins.append(d)
            yt=np.array(yt);yp=np.array(yp);yn=np.array(yn);base=np.array(base)
            if dates is not None:                                  # >>> FIX: xuat preds Chronos (ngay MUC TIEU d+h)
                dts=pd.to_datetime(np.asarray(dates)[np.array(origins)+h])
                save_preds("Chronos", h, dts, yt, yp, outdir=PREDS_DIR)
            mm=float(np.mean(np.abs(yt-yp)));nm=float(np.mean(np.abs(yt-yn)))
            rm=float(np.sqrt(np.mean((yt-yp)**2)))
            da=float(np.mean(np.sign(yt-base)==np.sign(yp-base)))
            dm,p=dm_test(yt-yp,yt-yn,h=h)
            rows.append(dict(model="Chronos",h=h,n=int(len(yt)),MAE=round(mm,1),
                             MAE_naive=round(nm,1),
                             dMAE_pct=round(100*(mm/nm-1),2) if nm>0 else None,
                             RMSE=round(rm,1),DA=round(da,3),DM=round(dm,3),DM_p=round(p,4)))
            print(f"  Chronos h={h:2d}: MAE={mm:9.1f} Naive={nm:9.1f} dMAE%={100*(mm/nm-1):+6.2f} "
                  f"DA={da:.3f} DM_p={p:.4f} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  => LOI h={h}: {type(e).__name__}: {e}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--horizons",type=int,nargs="+",default=HORIZONS)
    ap.add_argument("--skip_dl",action="store_true")
    ap.add_argument("--skip_chronos",action="store_true")
    ap.add_argument("--context",type=int,default=512)
    ap.add_argument("--refit_every",type=int,default=1,
                help="1=TRUE expanding: refit moi origin (dung protocol, cham); "
                     "k>1 = refit moi k origin; 0 = fit 1 lan (pseudo, nhanh)")
    a=ap.parse_args()
    print(f"[{_now()}] Load..."); df=load_data(); serie=df[TARGET].astype(float).values
    rows=[]
    if not a.skip_dl: run_nhits_nbeatsx(df, a.horizons, rows, a.refit_every)
    if not a.skip_chronos: run_chronos_all(serie, a.horizons, rows, context=a.context, dates=df[DATE_COL].values)
    out=pd.DataFrame(rows)
    os.makedirs("E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\", exist_ok=True)
    out.to_csv("E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\dl_comparison.csv", index=False)
    try:
        stat=pd.read_csv("E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\TS_model\\ts_model_comparison.csv")
        pd.concat([stat,out],ignore_index=True).sort_values(["h","MAE"]).to_csv("E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\all_comparison.csv",index=False)
    except FileNotFoundError:
        out.to_csv("E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\all_comparison.csv", index=False)
    print("\n[done] -> E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\dl_comparison.csv , E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\DL_model\\all_comparison.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
