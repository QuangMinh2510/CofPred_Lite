# CofPred full-stack dashboard

Thư mục này là web app chính theo kiến trúc trong slide:

```text
React 18 + TypeScript + ECharts
              ↓ HTTP/JSON
FastAPI + Pydantic + SQLAlchemy
              ↓
SQLite (chạy nhanh) hoặc TimescaleDB + Redis (Docker)
              ↓
data/processed/*.csv + models/model_registry.csv
```

Web phân biệt hai lớp kết quả:

- **Research:** 22 model, 4 horizon, walk-forward và MCS trong `scripts/` và `results/`.
- **Deployment:** chỉ model/horizon thực sự có trong `models/model_registry.csv`.

Nếu horizon chưa có model deploy, API dùng RandomWalk fallback và ghi rõ lý do; không tạo dự báo giả.

## Sáu panel

1. Giá và fan chart 50/80/95%.
2. Tín hiệu LONG/FLAT tại `h=5`.
3. So sánh MSE và Δ% của 22 model với RandomWalk.
4. Giá nội địa, London quy đổi và basis.
5. USD/VND, diesel, ONI, mưa và cân bằng nước.
6. MAE/RMSE/DA của model deploy và kết luận MCS nghiên cứu.

## Chạy nhanh trong VS Code

Yêu cầu Python 3.11/3.12 và Node.js 20.

### Terminal 1 — Backend

Chạy từ thư mục gốc repository:

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r webapp/backend/requirements.txt
& ".\.venv\Scripts\python.exe" -m uvicorn webapp.backend.app.main:app --reload --port 8000
```

Kiểm tra API:

- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

### Terminal 2 — Frontend

```powershell
cd webapp/frontend
npm install
npm run dev
```

Mở <http://localhost:5173>.

SQLite được dùng mặc định nên không cần cài database để demo.

## Chạy đầy đủ bằng Docker

Chạy từ thư mục `webapp`:

```bash
docker compose up --build
```

- Dashboard: <http://localhost:8080>
- API: <http://localhost:8000/docs>
- TimescaleDB: `localhost:5432`
- Redis: `localhost:6379`

## API chính

| Endpoint | Chức năng |
|---|---|
| `/api/meta` | Phạm vi dữ liệu, research và model deploy |
| `/api/prices` | Lịch sử giá nội địa |
| `/api/forecast` | Point forecast và dải 50/80/95% |
| `/api/signal` | LONG/FLAT h=5 |
| `/api/model-comparison` | Bảng 22 model theo horizon |
| `/api/basis` | Nội địa, London và basis |
| `/api/drivers` | Macro drivers dạng z-score |
| `/api/reliability` | Metric deploy và kết luận MCS |
| `/api/mcs` | Tóm tắt MCS offline |
| `/api/ingestion-status` | Trạng thái nạp dữ liệu |

## Cập nhật dữ liệu theo lịch

Chạy một lần:

```powershell
& ".\.venv\Scripts\python.exe" -m webapp.backend.app.workers.scheduler --once
```

Chạy lịch hằng ngày lúc 18:30:

```powershell
& ".\.venv\Scripts\python.exe" -m webapp.backend.app.workers.scheduler
```

Mọi dự báo chỉ hỗ trợ quyết định và học tập, không phải khuyến nghị đầu tư.

