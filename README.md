# OptimAIze

Một workspace cha (parent) theo kiến trúc **module độc lập, kích hoạt tường minh**: lớp điều phối mỏng quản lý các module con chạy được độc lập. Module con đầu tiên là **OptimAIze-OCR** — pipeline OCR layout-aware tối ưu cho CPU (không cần GPU), gắn vào qua git submodule.

> Tài liệu chi tiết: [ARCHITECTURE.md](ARCHITECTURE.md) · link model + benchmark: [BENCHMARKS.md](BENCHMARKS.md)

---

## Thành phần

| Thành phần | Vai trò | Cổng |
|---|---|---|
| Parent Gradio UI | UI điều phối, xem trạng thái module | 7850 |
| Parent API (FastAPI) | Liệt kê module, launch UI con | 8000 |
| Parent Web (React + Vite) | Dashboard workspace | 5174 |
| OCR API (submodule) | OCR thật + lịch sử (SQLite) | 8001 |
| OCR Web (submodule) | UI workspace OCR | 5173 |

Parent **không tự chứa engine AI** — nó bắc cầu (bridge) tới OptimAIze-OCR. Model nặng chỉ load sau hành động tường minh của người dùng (lazy loading).

---

## Yêu cầu

- **Python 3.12** (khuyến nghị — stack OCR pin `torch`/`transformers`/`numba`).
- **Node.js** (cho frontend React/Vite).
- (Tùy chọn) compiler C++ với AVX2 (g++/clang++/MSVC) để bật kernel AVX2; thiếu thì tự fallback Numba JIT.

## Cài đặt

```bash
# Clone cả submodule
git clone --recurse-submodules https://github.com/BigM219/OptimAIze.git
cd OptimAIze

# venv Python 3.12
py -3.12 -m venv .venv

# Parent (nhẹ — chạy UI được ngay cả khi chưa cài OCR)
.venv/Scripts/python.exe -m pip install -e .

# OCR submodule (nặng — torch, transformers... vài GB)
.venv/Scripts/python.exe -m pip install -e modules/OptimAIze-OCR

# Frontend
cd apps/parent-web && npm install
cd modules/OptimAIze-OCR/apps/web && npm install
```

> Nếu submodule trống sau khi clone: `git submodule update --init --recursive`.

## Chạy

```bash
# Parent Gradio UI (7850)
python main.py

# Parent API (8000)
uvicorn optimaize_parent_api.main:app --app-dir apps/parent-api --port 8000

# Parent web (5174)
cd apps/parent-web && npm run dev

# --- OCR (độc lập) ---
# OCR API (8001)
uvicorn optimaize_ocr_api.main:app --app-dir modules/OptimAIze-OCR/apps/api --port 8001

# OCR web (5173)
cd modules/OptimAIze-OCR/apps/web && npm run dev

# OCR qua Docker (api 8001 + web 8080)
cd modules/OptimAIze-OCR && docker compose up
```

## Test

```bash
.venv/Scripts/python.exe -m pytest tests/ \
  apps/parent-api/optimaize_parent_api/tests/ \
  modules/OptimAIze-OCR/apps/api/optimaize_ocr_api/tests/
```

---

## Cấu hình môi trường

### Quản trị tài nguyên (chống treo)

OCR chạy trên CPU; trên máy hạn chế, oversubscription luồng hoặc vượt RAM có thể làm treo server. Lớp resource governance suy ra trần luồng + RAM theo phần cứng thật và profile:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OPTIMAIZE_ENV` | `dev` | `dev` → mục tiêu ~50% tài nguyên; `prod` → ~80% |

- **dev**: thread cap = 50% nhân vật lý, RAM cap = 50% tổng RAM.
- **prod**: thread cap = 80% nhân vật lý, RAM cap = 80% tổng RAM.
- Model ước tính vượt trần RAM bị **từ chối (HTTP 503)** kèm thông báo, thay vì để swap-thrash dẫn tới treo.

### Bảo mật API

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OPTIMAIZE_API_KEY` | (trống) | Đặt để bật auth: client phải gửi header `X-API-Key`. Không đặt → auth tắt (tiện cho local). `/health` luôn mở. |
| `OPTIMAIZE_PARENT_CORS_ORIGINS` | localhost dev | Danh sách CORS origin (CSV) cho parent API |
| `OPTIMAIZE_OCR_CORS_ORIGINS` | localhost dev | Danh sách CORS origin (CSV) cho OCR API |

> ⚠️ Khi `OPTIMAIZE_API_KEY` chưa đặt, API mở hoàn toàn — phù hợp chạy local, **không nên expose ra mạng** nếu chưa đặt key và siết CORS.

### Đường dẫn & lưu trữ (OCR)

| Biến | Ý nghĩa |
|---|---|
| `OPTIMAIZE_OCR_OUTPUT_DIR` | Thư mục output |
| `OPTIMAIZE_OCR_UPLOAD_DIR` | Thư mục upload |
| `OPTIMAIZE_OCR_HISTORY_DB` | Đường dẫn SQLite history |
| `OPTIMAIZE_OCR_WEIGHTS_DIR` | Thư mục cache weights quantized |
| `OPTIMAIZE_PROJECT_ROOT` | Ghi đè vị trí workspace root (dùng bởi bridge) |

---

## Cấu trúc thư mục

```
OptimAIze/
├── main.py                       # entry point Gradio UI parent
├── packages/parent-core/         # lõi: app_ui.py + modules/ocr_bridge.py (cầu nối)
├── apps/parent-api/              # FastAPI parent
├── apps/parent-web/              # React + Vite parent
└── modules/OptimAIze-OCR/        # submodule: engine OCR + API + web + Docker
```

Xem [ARCHITECTURE.md](ARCHITECTURE.md) để hiểu luồng xử lý OCR, stack tối ưu CPU (INT8 + AVX2 + fused kernels), và luồng API/frontend.

---

## Module OCR

OptimAIze-OCR chạy `PP-DocLayoutV3` để phát hiện cấu trúc tài liệu, crop từng vùng, rồi đưa qua một backend Vision-Language Model. Hỗ trợ nhiều backend: `falcon-ocr`, `lighton-ocr`, `dots-mocr`, `paddleocr-vl`, `glm-ocr`, `surya-ocr`, `surya-package`.

Link model HuggingFace và benchmark tham khảo: [BENCHMARKS.md](BENCHMARKS.md). Tài liệu engine chi tiết: [README submodule](modules/OptimAIze-OCR/README.md).
