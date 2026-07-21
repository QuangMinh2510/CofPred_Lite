# CofPred — Vietnamese Robusta Coffee Price Forecasting

CofPred gồm hai phần được tách rõ để kết quả nghiên cứu không bị nhầm với model đang triển khai:

1. **Research pipeline** (`scripts/`) so sánh 22 mô hình trên bốn horizon `1/5/21/63`, dùng walk-forward và Model Confidence Set.
2. **Full-stack dashboard** (`webapp/`) dùng FastAPI, React, TypeScript và ECharts để trình bày dữ liệu, dự báo, tín hiệu và độ tin cậy.

Kết luận nghiên cứu chính: **không có bằng chứng thống kê đủ mạnh rằng mô hình phức tạp đánh bại RandomWalk**. Vì vậy dashboard luôn giữ RandomWalk làm benchmark và không biến một MSE thấp hơn thành tuyên bố thắng có ý nghĩa thống kê.

## Cấu trúc repository

```text
CofPred_Lite/
├── data/                         # raw, processing và master CSV
├── models/                       # model .joblib + model_registry.csv
├── results/                      # prediction, metric và kết quả nghiên cứu
├── scripts/
│   ├── crawl/                    # thu thập dữ liệu
│   ├── process/                  # làm sạch và tạo master/features
│   ├── model/                    # pipeline 22 model
│   └── analysis/                 # MCS, kiểm định và backtest
├── webapp/
│   ├── backend/
│   │   ├── app/                  # FastAPI, SQLAlchemy, Pydantic, scheduler
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── frontend/
│   │   ├── src/                  # React 18 + TypeScript + ECharts
│   │   ├── package.json
│   │   └── Dockerfile
│   ├── docker-compose.yml        # TimescaleDB, Redis, API, scheduler, UI
│   └── README.md
├── app.py                        # dashboard Streamlit cũ, giữ để đối chiếu
├── live_data.py
└── train_models.py
```

## Sáu panel của dashboard

1. **Price + forecast fan chart:** giá lịch sử, point forecast và dải 50/80/95%.
2. **Direction signal:** LONG/FLAT tại `h=5`; dùng RandomWalk/FLAT nếu chưa có model h=5 deploy.
3. **22-model comparison:** MSE và Δ% so với RandomWalk tại từng horizon.
4. **Domestic vs London basis:** giá nội địa, London quy đổi và chênh lệch.
5. **Macro drivers:** USD/VND, diesel, ONI, mưa 90 ngày và cân bằng nước dạng z-score.
6. **Model reliability:** MAE/RMSE/DA của model deploy và kết luận MCS offline.

## Chạy trong VS Code

Yêu cầu Python 3.11/3.12 và Node.js 20.

### Backend — Terminal 1

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r webapp/backend/requirements.txt
& ".\.venv\Scripts\python.exe" -m uvicorn webapp.backend.app.main:app --reload --port 8000
```

Swagger: <http://localhost:8000/docs>

### Frontend — Terminal 2

```powershell
cd webapp/frontend
npm install
npm run dev
```

Dashboard: <http://localhost:5173>

Mặc định backend dùng SQLite nên có thể demo ngay, không cần cài PostgreSQL.

## Chạy đầy đủ bằng Docker

```bash
cd webapp
docker compose up --build
```

- Dashboard: <http://localhost:8080>
- API: <http://localhost:8000/docs>
- TimescaleDB: `localhost:5432`
- Redis: `localhost:6379`

## Phân biệt research và deployment

- `scripts/model/` và `scripts/analysis/`: 22 model, walk-forward, MCS — chạy offline.
- `models/model_registry.csv`: danh sách model/horizon thật sự đang deploy.
- Nếu registry chưa có một horizon, API trả RandomWalk fallback và lý do cụ thể.
- Không được nói web đang deploy 22 model chỉ vì bảng nghiên cứu có 22 model.

## API

`/api/health` · `/api/meta` · `/api/prices` · `/api/forecast` · `/api/signal` · `/api/model-comparison` · `/api/basis` · `/api/drivers` · `/api/reliability` · `/api/mcs` · `/api/ingestion-status`

Chi tiết vận hành xem [`webapp/README.md`](webapp/README.md).

> CofPred là công cụ hỗ trợ quyết định và học tập, không phải khuyến nghị đầu tư.

