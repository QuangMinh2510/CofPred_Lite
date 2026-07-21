# Neu chay tren Colab/may chua co thu vien, bo comment dong duoi:
# !pip install scikit-learn scipy statsmodels xgboost lightgbm

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')   # backend khong GUI, phai goi TRUOC khi import pyplot
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor


HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

HAS_LGB = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGB = False

HAS_SM = True
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    HAS_SM = False

plt.rcParams['figure.figsize'] = (12, 5)
print('xgboost:', HAS_XGB, '| lightgbm:', HAS_LGB, '| statsmodels:', HAS_SM)

import sys
sys.path.insert(0, r"E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\scripts\\model")  # noi de export_preds.py
from export_preds import save_preds
PREDS_DIR = r"E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\preds"


CSV_PATH = 'E:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\processed\gia_cafe_master_full.csv'
df = pd.read_csv(CSV_PATH, parse_dates=['Ngay']).sort_values('Ngay').reset_index(drop=True)
df = df.set_index('Ngay')
print('Shape:', df.shape)
print('Khoang thoi gian:', df.index.min().date(), '->', df.index.max().date())

# --- KIEM TRA CHONG NHAM FILE CU (bi ro ri) ---
n_weekend = int((df.index.dayofweek >= 5).sum())
if n_weekend > 0:
    print(f"\n[CANH BAO] Co {n_weekend} dong cuoi tuan (T7/CN) -> day la MASTER CU bi ro ri!")
    print("           Hay chay lai: gia_taynguyen.py + 41_build_master_full.py de ghi de file.")
    print("           File dung phai co ~1566 dong, 0 dong cuoi tuan.")
else:
    print(f"\n[OK] Khong co ngay cuoi tuan. So dong: {len(df)} (ky vong ~1566).")
df.head()



TARGET = 'Gia_target'

# Bo 24 feature day du khop mo hinh huong 66,1% trong bao cao
AR = ['target_lag1','target_lag2','target_lag3','target_ret_lag1',
      'MA5','MA10','std5','dayofweek','month']
EXO = ['london_vnd_kg_lag1','usdvnd_lag1','diesel','diesel_chg_1m','diesel_chg_3m',
       'Luong_lag1m','rain_90d','oni','waterbal_90d']
SUPPLY = ['area_tn','prod_tn','yield_tn','tonkho_tan','dongia_lag1m','dongia_ret_lag1m']
WANT = AR + EXO + SUPPLY

# Chi giu cot thuc su co trong file; canh bao cot thieu
FEATURES = [c for c in WANT if c in df.columns]
missing = [c for c in WANT if c not in df.columns]
if missing:
    print('[CANH BAO] Thieu', len(missing), 'feature trong file:', missing)
    print('           -> Hay chay lai 41_build_master_full.py de sinh du 24 feature.')
print('Dung', len(FEATURES), 'feature:', FEATURES)

data = df[[TARGET] + FEATURES].copy()

# prev_price (gia ngay truoc) dung cho Naive va delta mode; KHONG nam trong FEATURES
prev_price = data[TARGET].shift(1).rename('prev_price')
data['prev_price'] = prev_price

before = len(data)
data = data.dropna(subset=[TARGET] + FEATURES + ['prev_price'])
print('Bo', before - len(data), 'dong NaN (warm-up + dut bao gia) ->', len(data), 'dong dung duoc')

X = data[FEATURES].copy()
prev_series = data['prev_price'].copy()
y_level = data[TARGET].copy()
print('X:', X.shape, '| y:', y_level.shape)



# === CHON CHAN TROI DU BAO ===
# HORIZON = so ngay LAM VIEC du bao toi:
#   1  = ngay mai        5  = 1 tuan        10 = 2 tuan
#   21 = 1 thang         63 = 1 quy
# Doi gia tri nay roi chay lai tu cell nay tro xuong de co ket qua tung chan troi.
# HORIZON = 1
# TARGET_MODE = 'level'   # 'delta' (du bao thay doi gia) hoac 'level'
for HORIZON in [1, 5, 21, 63]:
    for TARGET_MODE in ['delta', 'level']:

        _HN = {1:'1 ngay (ngay mai)',5:'1 tuan',10:'2 tuan',21:'1 thang',63:'1 quy'}
        HORIZON_NAME = _HN.get(HORIZON, f'{HORIZON} ngay lam viec')

        price  = y_level.copy()            # gia(t) - da biet tai thoi diem t
        future = price.shift(-HORIZON)     # gia(t+h) - dai luong CAN du bao
        anchor = price.copy()              # moc hien tai = gia(t)

        print(f'HORIZON = {HORIZON} ngay lam viec ({HORIZON_NAME}) | TARGET_MODE = {TARGET_MODE}')
        print(f'Du bao gia(t+{HORIZON}) tu thong tin biet den ngay t (KHONG phai ngay mai).')




        def mae(a, p):  return float(np.mean(np.abs(a - p)))
        def rmse(a, p): return float(np.sqrt(np.mean((a - p) ** 2)))

        def smape(a, p):
            denom = (np.abs(a) + np.abs(p)) / 2.0
            return float(np.mean(np.abs(a - p) / np.where(denom == 0, np.nan, denom)) * 100)

        def mase(a, p, y_train):
            # SUA: mau so = MAE naive tren PHAN TRAIN BAN DAU (khong gom test).
            # Ban cu dung toan chuoi (ca test) -> lech chuan MASE.
            d = np.mean(np.abs(np.diff(y_train)))
            return float(mae(a, p) / d)

        def directional_accuracy(a, p, prev):
            # ti le du bao dung huong tang/giam so voi ngay truoc
            return float(np.mean(np.sign(a - prev) == np.sign(p - prev)) * 100)




        def walk_forward(make_model, X, future, anchor, initial=0.6, step=5, horizon=5, mode='delta'):
            # Walk-forward expanding window. Tai moi diem quyet dinh d:
            #   - du bao gia(t+h) = gia(d+horizon)
            #   - TRAIN chi gom cac cap (t, t+h) da hoan thanh: t+h <= d  <=>  t <= d-horizon
            #     -> nhan huan luyen khong nhin tuong lai.
            n = len(X)
            start = int(n * initial)
            recs = []
            for d in sorted(range(n - 1 - horizon, start - 1, -step)):
                tr = d - horizon + 1                 # so dong train hop le (t = 0..d-horizon)
                if tr <= 20:
                    continue
                Xtr = X.iloc[:tr]
                ytr = (future - anchor).iloc[:tr] if mode == 'delta' else future.iloc[:tr]
                m = make_model(); m.fit(Xtr, ytr)
                raw = float(m.predict(X.iloc[[d]])[0])
                pred = float(anchor.iloc[d]) + raw if mode == 'delta' else raw
                recs.append((X.index[d], float(future.iloc[d]), float(anchor.iloc[d]), pred))
            return pd.DataFrame(recs, columns=['Ngay','y_true','anchor','pred']).set_index('Ngay')



        def make_models(horizon=1):
            # === Hyperparameter duoc tune rieng cho tung chan troi ===
            # h=1  (1 ngay) : autocorrelation cao, signal manh -> regularization nho, model manh
            # h=5  (1 tuan) : AR signal con kha     -> regularization vua
            # h=21 (1 thang): AR signal yeu dan     -> regularization tang, don gian hon
            # h=63 (1 quy)  : noise cao, exo/supply quan trong -> regularization manh, tranh overfit
            _cfg = {
                1:  dict(ridge_a=0.5,  lasso_a=0.0005, enet_a=0.0005, enet_l1=0.3,
                         rf_est=400, rf_leaf=2,  rf_depth=None,
                         xgb_est=600, xgb_lr=0.02, xgb_depth=5, xgb_sub=0.8, xgb_col=0.8,
                         lgb_est=700, lgb_lr=0.02, lgb_leaves=40,
                         svr_C=20,  svr_eps=0.05,
                         knn_k=5),
                5:  dict(ridge_a=1.0,  lasso_a=0.001,  enet_a=0.001,  enet_l1=0.5,
                         rf_est=400, rf_leaf=3,  rf_depth=None,
                         xgb_est=500, xgb_lr=0.03, xgb_depth=4, xgb_sub=0.8, xgb_col=0.8,
                         lgb_est=600, lgb_lr=0.03, lgb_leaves=31,
                         svr_C=10,  svr_eps=0.1,
                         knn_k=10),
                21: dict(ridge_a=5.0,  lasso_a=0.01,   enet_a=0.01,   enet_l1=0.5,
                         rf_est=500, rf_leaf=5,  rf_depth=10,
                         xgb_est=400, xgb_lr=0.05, xgb_depth=3, xgb_sub=0.7, xgb_col=0.7,
                         lgb_est=500, lgb_lr=0.05, lgb_leaves=20,
                         svr_C=5,   svr_eps=0.5,
                         knn_k=15),
                63: dict(ridge_a=10.0, lasso_a=0.1,    enet_a=0.1,    enet_l1=0.7,
                         rf_est=500, rf_leaf=10, rf_depth=8,
                         xgb_est=300, xgb_lr=0.05, xgb_depth=3, xgb_sub=0.7, xgb_col=0.7,
                         lgb_est=400, lgb_lr=0.05, lgb_leaves=15,
                         svr_C=1,   svr_eps=1.0,
                         knn_k=20),
            }
            c = _cfg.get(horizon, _cfg[5])  # fallback ve h=5 neu horizon la, khac
            m = {}
            m['Ridge']      = lambda c=c: Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=c['ridge_a']))])
            m['Lasso']      = lambda c=c: Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=c['lasso_a'], max_iter=50000))])
            m['ElasticNet'] = lambda c=c: Pipeline([('sc', StandardScaler()), ('m', ElasticNet(alpha=c['enet_a'], l1_ratio=c['enet_l1'], max_iter=50000))])
            m['RandomForest'] = lambda c=c: RandomForestRegressor(
                n_estimators=c['rf_est'], min_samples_leaf=c['rf_leaf'],
                max_depth=c['rf_depth'], n_jobs=-1, random_state=42)
            if HAS_XGB:
                m['XGBoost'] = lambda c=c: XGBRegressor(
                    n_estimators=c['xgb_est'], learning_rate=c['xgb_lr'], max_depth=c['xgb_depth'],
                    subsample=c['xgb_sub'], colsample_bytree=c['xgb_col'], random_state=42, n_jobs=-1)
            if HAS_LGB:
                m['LightGBM'] = lambda c=c: LGBMRegressor(
                    n_estimators=c['lgb_est'], learning_rate=c['lgb_lr'], num_leaves=c['lgb_leaves'],
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
            m['SVR'] = lambda c=c: TransformedTargetRegressor(
                regressor=Pipeline([('sc', StandardScaler()),
                        ('m', SVR(C=c['svr_C'], epsilon=c['svr_eps'], kernel='rbf'))]),
                transformer=StandardScaler())
            m['kNN'] = lambda c=c: Pipeline([('sc', StandardScaler()), ('m', KNeighborsRegressor(n_neighbors=c['knn_k']))])
            return m

        MODELS = make_models(horizon=HORIZON)
        print(f'Models (h={HORIZON}):', list(MODELS.keys()))




        INITIAL, STEP = 0.6, 5      # STEP = nhip lay diem test (giam chong lap), KHONG phai chan troi
        results_blocks = {}
        n = len(X); start = int(n * INITIAL)

        # --- Naive (random walk): du bao gia(t+h) = gia(t) hien tai (khong doi) ---
        naive_rows = []
        for d in sorted(range(n - 1 - HORIZON, start - 1, -STEP)):
            naive_rows.append((X.index[d], float(future.iloc[d]), float(anchor.iloc[d]), float(anchor.iloc[d])))
        naive_df = pd.DataFrame(naive_rows, columns=['Ngay','y_true','anchor','pred']).set_index('Ngay')
        results_blocks['Naive'] = naive_df

        EVAL    = naive_df.index                 # truc thoi gian danh gia chung
        y_true  = naive_df['y_true'].values      # = gia(t+h) thuc te
        anchor_eval = naive_df['anchor'].values  # = gia(t)
        print(f'So diem test (chan troi {HORIZON_NAME}): {len(EVAL)}')

        # --- ARIMAX(1,1,1): fit den ngay t, du bao h buoc, lay buoc thu h ---
        # Exog giu co dinh = gia tri ngay t (tranh dung exog tuong lai -> khong look-ahead).
        if HAS_SM:
            exog_cols = [c for c in EXO if c in FEATURES]
            arx_rows = []
            for d in sorted(range(n - 1 - HORIZON, start - 1, -STEP)):
                tr = d + 1
                try:
                    r = SARIMAX(price.iloc[:tr], exog=X[exog_cols].iloc[:tr], order=(1,1,1),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    exog_future = np.tile(X[exog_cols].iloc[[d]].values, (HORIZON, 1))
                    fc = np.asarray(r.forecast(steps=HORIZON, exog=exog_future))
                    pred = float(fc[-1])
                except Exception:
                    pred = float(anchor.iloc[d])
                arx_rows.append((X.index[d], float(future.iloc[d]), float(anchor.iloc[d]), pred))
            results_blocks['ARIMAX(1,1,1)'] = pd.DataFrame(
                arx_rows, columns=['Ngay','y_true','anchor','pred']).set_index('Ngay')
        else:
            print('Bo qua ARIMAX (chua co statsmodels)')




        for name, factory in MODELS.items():
            results_blocks[name] = walk_forward(
                factory, X, future, anchor,
                initial=INITIAL, step=STEP, horizon=HORIZON, mode=TARGET_MODE)
        print('Da chay:', list(results_blocks.keys()))



        # MASE: mau so = MAE naive 1-buoc tren PHAN TRAIN BAN DAU (khong gom test)
        y_train_mase = price.values[:int(len(price) * INITIAL)]
        rows = []
        for name, blk in results_blocks.items():
            b = blk.reindex(EVAL)
            p = b['pred'].values.astype(float)
            a = b['y_true'].values.astype(float)
            anc = b['anchor'].values.astype(float)
            msk = np.isfinite(p) & np.isfinite(a)
            rows.append({
                'Model': name,
                'MAE':   mae(a[msk], p[msk]),
                'RMSE':  rmse(a[msk], p[msk]),
                'sMAPE(%)': smape(a[msk], p[msk]),
                'MASE':  mase(a[msk], p[msk], y_train_mase),
                'DA(%)': directional_accuracy(a[msk], p[msk], anc[msk]),
            })
        res = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
        os.makedirs("E:\\FPT\AI\\SEM8_AI\\DAP391m\\project\CofPred\\results\\ML_model", exist_ok=True)
        res.to_csv(f'E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\ML_model\\model_comparison_h{HORIZON}_{TARGET_MODE}.csv', index=False)
        print(f'=== Chan troi {HORIZON_NAME} (h={HORIZON}, n={len(EVAL)} diem test) ===')
        print(res.to_string())



        from export_preds import save_preds
        idx = list(price.index)                          # DatetimeIndex chuoi gia da align (1516)
        posmap = {dd: i for i, dd in enumerate(idx)}
        for name, blk in results_blocks.items():
            if name == 'SVR' and TARGET_MODE == 'level':
                continue    # SVR-RBF chi giu preds mode 'delta' (level bi tru dep do khong ngoai suy)
            b = blk.reindex(EVAL).dropna(subset=['pred'])
            tdates = [idx[posmap[od] + HORIZON] for od in b.index]
            save_preds(name, HORIZON, tdates, b['y_true'], b['pred'], anchor=b['anchor'], outdir=PREDS_DIR)

        best = res.iloc[0]['Model']   # lay best model sau khi vong for ket thuc
        bb = results_blocks[best].reindex(EVAL)
        nn = results_blocks['Naive'].reindex(EVAL)
        plt.figure(figsize=(13,5))
        plt.plot(EVAL, y_true,        label='Thuc te gia(t+h)', color='black', lw=1.2)
        plt.plot(EVAL, bb['pred'],    label='Du bao (' + best + ')', color='tab:red', lw=1)
        plt.plot(EVAL, nn['pred'],    label='Naive', color='tab:blue', lw=0.7, alpha=0.6)
        plt.title(f'Du bao vs thuc te - {best} - chan troi {HORIZON_NAME} (mode={TARGET_MODE})')
        plt.ylabel('Gia (VND/kg)'); plt.legend(); plt.tight_layout()
        plt.savefig(f'E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\ML_model\\forecast_best_h{HORIZON}_{TARGET_MODE}.png', dpi=120)
        plt.close()   # giai phong memory, khong goi tkinter



        # LUU Y: cell nay fit tren TOAN BO du lieu -> chi de XEM do quan trong feature,
        # KHONG dung de chon feature (selection bias).
        tree_pref = [m for m in ['LightGBM','XGBoost','RandomForest'] if m in MODELS]
        imp_name = next((c for c in tree_pref if c in res['Model'].values), 'RandomForest')

        yfull = (future - anchor) if TARGET_MODE == 'delta' else future
        msk = future.notna()
        mdl = MODELS[imp_name]()
        mdl.fit(X[msk], yfull[msk])
        imp = getattr(mdl, 'feature_importances_', None)
        if imp is not None:
            fi = pd.Series(imp, index=FEATURES).sort_values()
            fi.plot(kind='barh', figsize=(8,7), title=f'Feature importance - {imp_name} (h={HORIZON})')
            plt.tight_layout(); plt.savefig(f'E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\ML_model\\feature_importance_h{HORIZON}_{TARGET_MODE}.png', dpi=120); plt.close()
            print(fi.sort_values(ascending=False))
        else:
            print('Model', imp_name, 'khong co feature_importances_')



        def diebold_mariano(y, p1, p2, h=1):
            # loss = squared error; DM>0: p1 te hon p2. HAC/Newey-West lag = h-1.
            d = (y - p1) ** 2 - (y - p2) ** 2
            d = d[np.isfinite(d)]
            nn = len(d)
            var = np.var(d, ddof=0)
            for k in range(1, h):
                var += 2 * np.cov(d[k:], d[:-k])[0, 1]
            return float(d.mean() / np.sqrt(var / nn))

        bb = results_blocks[best].reindex(EVAL)['pred'].values.astype(float)
        pn = results_blocks['Naive'].reindex(EVAL)['pred'].values.astype(float)
        msk = np.isfinite(bb) & np.isfinite(pn)
        dm = diebold_mariano(y_true[msk], bb[msk], pn[msk], h=HORIZON)
        print(f'DM ({best} vs Naive) tai chan troi {HORIZON_NAME} = {round(dm,3)}')
        print('  |DM|>1.96 => khac biet co y nghia 5%; DM>0 => Naive tot hon best.')
