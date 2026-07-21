#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_preds.py  --  Helper de XUAT du bao tung diem cho MCS
===============================================================
Vi cac script hien tai chi luu CHI SO TONG HOP (MAE/RMSE...), MCS can du bao
TUNG DIEM. File nay cung 1 ham save_preds() de cam vao pipeline san co: moi
model/horizon xuat ra results/preds/<MODEL>__h<h>.csv (cot Ngay,y_true,pred[,anchor]).
Sau do chay 12_mcs.py.

--- CACH DUNG (chi them 1 dong sau khi da co du bao tung diem) ---

1) modeling_pipeline_multihorizon.ipynb (ML level suite):
   Ban da co `results_blocks[name]` (index=Ngay, cot y_true, anchor, pred) va bien EVAL.
   Sau vong danh gia moi horizon, them:
       from export_preds import save_preds
       for name, blk in results_blocks.items():
           b = blk.reindex(EVAL).dropna(subset=['pred'])
           save_preds(name, H, b.index, b['y_true'], b['pred'], anchor=b.get('anchor'))

2) ts_model.ipynb (Naive/AutoARIMA/SARIMAX/VECM):
   Trong walk_forward_eval, ban co mang y_true & yhat + ngay tuong ung. Sau moi model:
       from export_preds import save_preds
       save_preds(model_name, horizon, dates_eval, y_true_arr, yhat_arr)

3) run_dl_models.py (NHITS/NBEATSx/Chronos):
   Sau khi co cv_h (cot ds,y,<Model>) hoac mang du bao Chronos:
       from export_preds import save_preds
       save_preds('NHITS', h, cv_h['ds'], cv_h['y'], cv_h['NHITS'])
       save_preds('NBEATSx', h, cv_h['ds'], cv_h['y'], cv_h['NBEATSx'])
       # Chronos: save_preds('Chronos', h, dates, yt, yhat)

LUU Y QUAN TRONG (de MCS can chinh dung):
  - MOI model phai xuat tren CUNG mau COMMON va CUNG walk-forward (start, STEP)
    thi cot 'Ngay' moi trung nhau; MCS tu dong lay giao cac ngay chung.
  - 'Ngay' phai la ngay cua diem duoc du bao (o+h), khong phai ngay goc o.
"""
import os
import numpy as np
import pandas as pd


def save_preds(model, h, dates, y_true, y_pred, anchor=None, outdir='results/preds'):
    """Ghi results/preds/<model>__h<h>.csv voi cot Ngay,y_true,pred[,anchor]."""
    os.makedirs(outdir, exist_ok=True)
    d = pd.DataFrame({
        'Ngay': pd.to_datetime(np.asarray(dates)),
        'y_true': np.asarray(y_true, dtype=float),
        'pred': np.asarray(y_pred, dtype=float),
    })
    if anchor is not None:
        d['anchor'] = np.asarray(anchor, dtype=float)
    d = d.dropna(subset=['y_true', 'pred']).sort_values('Ngay').reset_index(drop=True)
    safe = str(model).replace('/', '-').replace(' ', '_')
    fp = os.path.join(outdir, f'{safe}__h{h}.csv')
    d.to_csv(fp, index=False)
    print(f'  [save_preds] {fp}  ({len(d)} diem)')
    return fp