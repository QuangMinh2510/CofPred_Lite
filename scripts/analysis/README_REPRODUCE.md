# CofPred - Bo script tai lap ket qua (bo sung)

Thu muc `analysis/` chua cac script tai lap **cac ket qua ma bao cao dang bao
nhung code goc chua co trong repo**: kiem dinh huong (logistic + Pesaran-Timmermann),
hieu chinh FDR, backtest gia tri kinh te, dong lien ket & truyen dan gia/ENSO.

## Cac script da co san (khong nam trong thu muc nay)
| Ket qua trong bai | File code goc |
|---|---|
| Thu thap & dung du lieu (18/24 dac trung) | `scripts/crawl/*`, `scripts/process/*` |
| Thong ke mo ta + ADF/KPSS | `thongke.py`, `ADF_KPSS.py` |
| Benchmark ML co dien (Bang 4, dong ML) | `modeling_pipeline*.ipynb` |
| Benchmark DL/foundation (Bang 4 + §5.4) | `run_dl_models*.py`, `ts_dl_model.ipynb` |

## Cac script BO SUNG (thu muc nay)
| # | File | Ket qua trong bai | Dau ra |
|---|---|---|---|
| 00 | `_config.py` | cau hinh chung (path, F18/F24, walk-forward) | - |
| 01 | `01_descriptive_stats.py` | §3.4 thong ke mo ta + tinh dung (tach tu thongke.py) | `results/descriptive_stats.csv`, `results/stationarity.csv` |
| 02 | `02_direction_test.py` | §5.2 kiem dinh huong logistic + Pesaran-Timmermann, F18 vs F24 | `results/direction_test.csv` |
| 03 | `03_fdr_correction.py` | §6 hieu chinh Benjamini-Hochberg (FDR) | `results/fdr_direction.csv` |
| 04 | `04_backtest.py` | §5.3 backtest Sharpe/MDD + permutation | `results/backtest.csv` (+ equity/perm) |
| 05 | `05_cointegration_transmission.py` | §5.5 Engle-Granger, ECM, Johansen/VECM, Gregory-Hansen, Granger/ENSO | `results/cointegration_summary.txt` |
| 06 | `06_figures.py` | 3 hinh minh hoa | `results/fig1_price.png`, `fig2_equity.png`, `fig3_permutation.png` |

## Cach chay
Chay tu THU MUC GOC cua repo (de path `data/processed/gia_cafe_master_full.csv` dung):
```bash
# moi truong: dung env 'cofpred' da co (pandas, scikit-learn, statsmodels, scipy)
python analysis/01_descriptive_stats.py
python analysis/02_direction_test.py
python analysis/03_fdr_correction.py
python analysis/04_backtest.py --horizons 5 10
python analysis/05_cointegration_transmission.py
python analysis/06_figures.py --break_date 2023-07-04 --horizon 5
```

## Luu y ve tai lap
- Cac script nay **cai dat DUNG phuong phap mo ta trong bai** (cung du lieu,
  cung bo dac trung, cung walk-forward chong look-ahead). Vi day la ban viet lai
  sach, con so co the LECH NHE so voi ket qua goc cua ban (do khac hat giong ngau
  nhien, phien ban thu vien, hoac chi tiet cau hinh logistic GD vs sklearn).
  Neu lech dang ke, hay gui lai output de doi chieu va tinh chinh.
- `02` dung `LogisticRegression` cua sklearn (L2) xap xi logistic-GD (lr0.2,
  iters800, l2=1e-3) trong bai. Neu muon khop tuyet doi, thay bang GD thu cong.
- Dataset `data/processed/gia_cafe_master_full.csv` KHONG kem trong goi code (du
  lieu tu xay dung). Xem muc "Data availability" trong bai bao.
