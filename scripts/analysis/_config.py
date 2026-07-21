# -*- coding: utf-8 -*-
"""
Cau hinh & tien ich dung chung cho cac script phan tich (CofPred).

Muc tieu: TAI LAP dung khung danh gia trung thuc trong bai bao
- Cung DATA_PATH, cung bo dac trung (F18/F24), cung walk-forward
  (INITIAL_FRAC=0.6, STEP=5, HORIZONS=[1,5,21,63]) nhu pipeline ML/DL.
- Khoa cung mau (align) -> N=1516 de moi bang so sanh dung 1 truc thoi gian.
"""
import numpy as np
import pandas as pd

DATA_PATH    = "data/processed/gia_cafe_master_full.csv"
TARGET       = "Gia_target"
INITIAL_FRAC = 0.60
STEP         = 5
HORIZONS     = [1, 5, 21, 63]
ANN          = 252            # so ngay giao dich / nam (annual hoa Sharpe)

# --- Bo dac trung (khop notebook modeling_pipeline_multihorizon) ---
AR     = ['target_lag1', 'target_lag2', 'target_lag3', 'target_ret_lag1',
          'MA5', 'MA10', 'std5', 'dayofweek', 'month']
EXO    = ['london_vnd_kg_lag1', 'usdvnd_lag1', 'diesel', 'diesel_chg_1m',
          'diesel_chg_3m', 'Luong_lag1m', 'rain_90d', 'oni', 'waterbal_90d']
SUPPLY = ['area_tn', 'prod_tn', 'yield_tn', 'tonkho_tan',
          'dongia_lag1m', 'dongia_ret_lag1m']
F18 = AR + EXO           # 18 dac trung (bo chinh thuc)
F24 = AR + EXO + SUPPLY   # 24 dac trung (mo rong, co bien cung)


def load_master(path=DATA_PATH):
    """Nap master, chuan hoa cot ngay thanh 'Ngay' va sap xep tang dan."""
    df = pd.read_csv(path)
    for c in ['date', 'Date', 'Ngay', 'ngay']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.sort_values(c).reset_index(drop=True)
            if c != 'Ngay':
                df = df.rename(columns={c: 'Ngay'})
            break
    return df


def align_frame(df, features):
    """Khoa cung mau: bo NaN theo TARGET + prev_price + features co san.
    Tra ve (df_da_loc, danh_sach_feature_co_san)."""
    d = df.copy()
    d['prev_price'] = d[TARGET].shift(1)
    have = [c for c in features if c in d.columns]
    missing = [c for c in features if c not in d.columns]
    if missing:
        print(f"[CANH BAO] Thieu {len(missing)} feature trong file: {missing}")
        print("           -> chay lai 41_build_master_full.py de sinh du dac trung.")
    d = d.dropna(subset=[TARGET, 'prev_price'] + have).reset_index(drop=True)
    print(f"[align] giu {len(d)} hang | start={int(len(d)*INITIAL_FRAC)} | features={len(have)}")
    return d, have


# ------------------------- Metric hoi quy -------------------------
def mae(a, p):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))
def rmse(a, p): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def max_drawdown(equity):
    """Max drawdown (so am) tu duong von tich luy (equity, dang x lan)."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())
