#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_ets_theta_combination.py
============================
Them 3 benchmark du bao kinh dien vao roster CofPred, dung THU VIEN CHUAN
(statsmodels) -- KHONG tu cai dat cong thuc -- tren DUNG khung walk-forward
neo-cuoi (last-anchored) lay tu scripts/analysis/_config.py (N=1516, STEP=5):

    ETS         : statsmodels ETSModel  ETS(A,Ad,N)
                  error='add', trend='add', damped_trend=True, seasonal=None
    Theta       : statsmodels ThetaModel (Assimakopoulos & Nikolopoulos 2000),
                  chuoi phi mua vu -> deseasonalize=False
    Combination : trung binh deu (Bates-Granger 1969) = 0.5*(ETS + Theta)

YEU CAU: statsmodels >= 0.12 (moi truong 'cofpred' cua ban da co san
         statsmodels/scipy/pandas/numpy).

----------------------------------------------------------------------------
CACH CHAY (tu THU MUC GOC repo, noi co data/processed/gia_cafe_master_full.csv):
    python scripts/model/add_ets_theta_combination.py

Ket qua ghi ra:  results/preds/{ETS,Theta,Combination}__h{1,5,21,63}.csv
                 (cot: Ngay,y_true,pred,anchor  -- dung format export_preds.py)

SAU DO cap nhat Model Confidence Set (dung 12_mcs.py da co san):
    for L in mae mse; do for h in 1 5 21 63; do \
        python scripts/analysis/12_mcs.py --preds results/preds --h $h --loss $L ; \
    done; done
----------------------------------------------------------------------------
GHI CHU: vi day dung uoc luong MLE cua statsmodels (khac ban numpy thu cong),
con so co the lech NHE so voi uoc luong so bo truoc do. Sau khi chay xong hay
chay lai 12_mcs.py de lay so MCS 22-model chinh thuc dua vao bai.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")   # tat warning hoi tu / khong co freq cho gon log

_HERE = os.path.dirname(os.path.abspath(__file__))
# dung chung cau hinh + ham align voi cac script phan tich khac
sys.path.insert(0, os.path.join(_HERE, "..", "analysis"))
sys.path.insert(0, _HERE)
from _config import (load_master, align_frame, TARGET,        # noqa: E402
                     INITIAL_FRAC, STEP, HORIZONS, F18)
from export_preds import save_preds                            # noqa: E402

from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa: E402
from statsmodels.tsa.forecasting.theta import ThetaModel        # noqa: E402

PREDS_DIR = "results/preds"


def forecast_ets(hist, h):
    """ETS(A,Ad,N) qua statsmodels; tra ve du bao tai buoc thu h."""
    model = ETSModel(np.asarray(hist, dtype=float),
                     error="add", trend="add", damped_trend=True,
                     seasonal=None, initialization_method="estimated")
    res = model.fit(disp=False)
    return float(np.asarray(res.forecast(h))[-1])


def forecast_theta(hist, h):
    """Theta method qua statsmodels; chuoi gia phi mua vu -> deseasonalize=False."""
    res = ThetaModel(np.asarray(hist, dtype=float),
                     period=1, deseasonalize=False).fit()
    return float(np.asarray(res.forecast(h))[-1])


def main():
    # ---- nap master + khoa cung mau (align) giong pipeline ML/DL ----
    df = load_master()
    d, have = align_frame(df, F18)
    n = len(d)
    start = int(n * INITIAL_FRAC)
    price = d.set_index("Ngay")[TARGET]
    yv = price.values.astype(float)
    idx = list(price.index)
    print(f"[align] n={n} | start={start} | "
          f"{'OK dung nhu bai (1516)' if n == 1516 else 'CHU Y: n != 1516'}")
    print(f"{'':<5}{'n':>5}{'MAE_Naive':>12}{'ETS':>12}{'Theta':>12}{'Combination':>13}")

    for h in HORIZONS:
        # ---- LUOI NEO-CUOI: y het modeling_pipeline_multihorizon.walk_forward ----
        origins = sorted(range(n - 1 - h, start - 1, -STEP))
        dates, y_true, anchor = [], [], []
        pE, pT, pC = [], [], []
        for o in origins:
            hist = yv[:o + 1]                 # cua so mo rong den moc o (chong leakage)
            e = forecast_ets(hist, h)
            t = forecast_theta(hist, h)
            pE.append(e); pT.append(t); pC.append(0.5 * (e + t))
            dates.append(idx[o + h])          # NGAY = diem duoc du bao (o + h)
            y_true.append(yv[o + h])          # gia thuc te tai (o + h)
            anchor.append(yv[o])              # moc hien tai gia(o)

        save_preds("ETS",         h, dates, y_true, pE, anchor=anchor, outdir=PREDS_DIR)
        save_preds("Theta",       h, dates, y_true, pT, anchor=anchor, outdir=PREDS_DIR)
        save_preds("Combination", h, dates, y_true, pC, anchor=anchor, outdir=PREDS_DIR)

        y_true = np.asarray(y_true, float)
        mae = lambda p: float(np.mean(np.abs(y_true - np.asarray(p, float))))
        nvp = os.path.join(PREDS_DIR, f"Naive__h{h}.csv")
        mn = (float(np.mean(np.abs((lambda x: x['y_true'] - x['pred'])(pd.read_csv(nvp)))))
              if os.path.exists(nvp) else float("nan"))
        print(f"h={h:<3d}{len(origins):5d}{mn:12,.1f}{mae(pE):12,.1f}{mae(pT):12,.1f}{mae(pC):13,.1f}")

    print("\nXong. Da ghi results/preds/{ETS,Theta,Combination}__h*.csv")
    print("Tiep theo: chay lai 12_mcs.py (mae & mse) de cap nhat MCS 22-model.")


if __name__ == "__main__":
    main()
