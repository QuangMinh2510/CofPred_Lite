# CofPred — Vietnamese Robusta Coffee Price Forecasting

Dự báo giá cà phê **Robusta nhân xô Tây Nguyên** — gồm hai phần:

1. **Research** — nghiên cứu học thuật so sánh 22 mô hình dự báo trên khung walk-forward
   chống look-ahead, dùng Model Confidence Set (MCS). Kết luận: *không mô hình phức tạp nào
   đánh bại random walk một cách có ý nghĩa thống kê.*
2. **Web app** (`webapp/`) — nền tảng market-intelligence: dashboard + API, dữ liệu cập nhật
   tự động, bán **bối cảnh quyết định + dải bất định trung thực** (không bán "độ chính xác").

> **Triết lý:** thà đúng và khiêm tốn còn hơn ấn tượng mà sai. Mọi dự báo đi kèm dải bất
> định thật và không phải khuyến nghị đầu tư.

---

## Kết quả nghiên cứu chính

- **Point forecast ≈ random walk:** ở cả 4 chân trời (1/5/21/63 ngày), không mô hình nào
  (kể cả LSTM, GRU, NHITS, NBEATSx, Chronos) đánh bại Naive có ý nghĩa (MCS + SPA đồng thuận).
- **DL bị loại đúng ở chân trời phân biệt tốt nhất** (h=1 dưới MAE; h=1, h=5 dưới MSE).
- **Giá trị kinh tế:** tín hiệu hướng h=5 sinh lời (+107%, Sharpe 1,1, permutation p=0,028)
  nhưng **phụ thuộc regime** (lỗ 2020–2023, lãi 2024–2025).

Bản thảo đầy đủ: `docs/paper/`.

---

## Cấu trúc thư mục

```
CofPred/
├── scripts/              # RESEARCH
│   ├── crawl/            #   thu thập dữ liệu (weather, fuel, usdvnd, ONI)
│   ├── process/          #   xử lý → data/processed/gia_cafe_master_full.csv
│   ├── model/            #   22 mô hình (walk-forward)
│   └── analysis/         #   MCS, SPA, DM, backtest, cointegration
├── data/                 # dữ liệu (raw / processed)
├── results/              # bảng so sánh, MCS, backtest, hình
├── docs/                 # bài báo + slide
└── webapp/               # WEB APP (tách biệt research)
    ├── backend/          #   FastAPI + SQLAlchemy (xem webapp/backend/README.md)
    ├── frontend/         #   React + Vite + ECharts (xem webapp/frontend/README.md)
    ├── ops/
    └── docker-compose.yml
```

Web app **không import** `scripts/` — nó giữ bản copy lõi model riêng trong
`webapp/backend/app/models/`; research chỉ dùng làm bootstrap dữ liệu một lần.

---

## Môi trường

Dùng conda env **`cofpred`** (Python 3.11) — không tạo venv riêng.

```bash
conda activate cofpred
pip install -r requirements.txt            # research deps
pip install -r webapp/backend/requirements.txt   # web deps (cài chung env)
```

---

## Chạy Web App

**1) Backend** (terminal 1, từ `webapp/backend`):
```bash
# nạp dữ liệu + dự báo lần đầu (một lần)
conda run -n cofpred python -m app.workers.scheduler --bootstrap
conda run -n cofpred python -m app.workers.scheduler --once --heavy

# chạy API
conda run -n cofpred python -m uvicorn app.main:app --port 8000
```

**2) Frontend** (terminal 2, từ `webapp/frontend`):
```bash
npm install       # lần đầu
npm run dev        # http://127.0.0.1:5173
```

**Tự động hàng ngày:** `conda run -n cofpred python -m app.workers.scheduler`
(crawl → build feature → dự báo → cập nhật DB, chạy 18:30 mỗi ngày).

### Endpoints chính
`/api/health` · `/api/prices` · `/api/forecast` · `/api/signal` · `/api/mcs` ·
`/api/backtest` · `/api/drivers` · `/api/ingestion-status` · `/docs` (OpenAPI)

---

## Pipeline dữ liệu

```
6 nguồn (crawl theo lịch) → raw_series → build_master → features
      → forecast 22 model + dải bất định + tín hiệu → forecasts/signals → API → Dashboard
```

| Nguồn | Dữ liệu | Cách lấy |
|---|---|---|
| baolamdong.vn | Giá Robusta TN (target) | server-render (requests+bs4) |
| Investing.com | London Robusta futures | curl-cffi |
| Yahoo Finance | USD/VND | yfinance |
| Open-Meteo | Mưa & bốc hơi 6 vùng | REST API |
| NOAA CPC | ENSO (ONI) | REST |
| luatvietnam.vn | Diesel | scraping |

---

## Công nghệ

**Backend:** FastAPI · SQLAlchemy 2.0 · Alembic · PostgreSQL+TimescaleDB (prod) / SQLite (dev) · Redis · APScheduler
**Frontend:** React 18 + TypeScript · Vite · Apache ECharts
**ML/DL:** scikit-learn · statsmodels · XGBoost · LightGBM · PyTorch + neuralforecast · Chronos (foundation model)
**Hạ tầng:** Docker Compose · Conda

---

## Tái lập kết quả nghiên cứu

Xem `scripts/analysis/README_REPRODUCE.md`. Chạy từ thư mục gốc repo (env `cofpred`):
```bash
python scripts/model/modeling_pipeline_multihorizon.py
python scripts/analysis/12_mcs.py --preds results/preds --h 5 --loss mae
python scripts/analysis/04_backtest.py --horizons 5 10
```

---

## Tác giả

Bao Trinh Tan Quang — AIT Laboratory, FPT University, Da Nang.
`baottqde170271@fpt.edu.vn`
