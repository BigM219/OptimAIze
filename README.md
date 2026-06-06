# OptimAIze

Một workspace cha (parent) theo kiến trúc **module độc lập, kích hoạt tường minh**: parent là lớp điều phối mỏng, mỗi module con chạy được độc lập (có source / UI / API / data riêng) và gắn vào qua git submodule.

Parent **không tự chứa engine AI**. Nó phát hiện module con, hiển thị trạng thái, và bắc cầu (bridge) sang module con — model nặng chỉ load sau hành động tường minh (lazy loading). Khi thêm module mới, parent chỉ cần biết cách dẫn sang nó.

> Tài liệu kiến trúc parent: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Module

| Module | Mô tả | Tài liệu |
|---|---|---|
| **OptimAIze-OCR** | Pipeline OCR layout-aware tối ưu cho CPU (không cần GPU) | [modules/OptimAIze-OCR/README.md](modules/OptimAIze-OCR/README.md) · [ARCHITECTURE](modules/OptimAIze-OCR/ARCHITECTURE.md) · [BENCHMARKS](modules/OptimAIze-OCR/BENCHMARKS.md) |

Mọi chi tiết về OCR (model, pipeline, tối ưu CPU, benchmark, cấu hình OCR) nằm trong chính module OptimAIze-OCR.

---

## Thành phần parent

| Thành phần | Vai trò | Cổng |
|---|---|---|
| Parent Gradio UI | UI điều phối, xem trạng thái module | 7850 |
| Parent API (FastAPI) | Liệt kê module, launch UI con | 8000 |
| Parent Web (React + Vite) | Dashboard workspace | 5174 |

---

## Yêu cầu

- **Python 3.12** (khuyến nghị).
- **Node.js** (cho frontend React/Vite).

## Cài đặt

```bash
# Clone cả submodule
git clone --recurse-submodules https://github.com/BigM219/OptimAIze.git
cd OptimAIze

# venv Python 3.12
py -3.12 -m venv .venv

# Parent (nhẹ — chạy UI được ngay cả khi chưa cài module con)
.venv/Scripts/python.exe -m pip install -e .

# Frontend parent
cd apps/parent-web && npm install
```

> Submodule trống sau khi clone? `git submodule update --init --recursive`.
> Cài đặt và chạy OptimAIze-OCR: xem [README của module](modules/OptimAIze-OCR/README.md).

## Chạy parent

```bash
# Parent Gradio UI (7850)
python main.py

# Parent API (8000)
uvicorn optimaize_parent_api.main:app --app-dir apps/parent-api --port 8000

# Parent web (5174)
cd apps/parent-web && npm run dev
```

## Test parent

```bash
.venv/Scripts/python.exe -m pytest tests/ apps/parent-api/optimaize_parent_api/tests/
```

---

## Cấu hình môi trường (parent)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OPTIMAIZE_API_KEY` | (trống) | Đặt để bật auth: client phải gửi header `X-API-Key`. Không đặt → auth tắt (tiện local). `/health` luôn mở. |
| `OPTIMAIZE_PARENT_CORS_ORIGINS` | localhost dev | Danh sách CORS origin (CSV) cho parent API |
| `OPTIMAIZE_PROJECT_ROOT` | (auto) | Ghi đè vị trí workspace root (dùng bởi bridge tới module con) |

> ⚠️ Khi `OPTIMAIZE_API_KEY` chưa đặt, API mở hoàn toàn — phù hợp local, **không nên expose ra mạng** nếu chưa đặt key và siết CORS.
> Cấu hình riêng của OCR (resource governance, paths, weights...) xem [tài liệu module](modules/OptimAIze-OCR/ARCHITECTURE.md#5-cấu-hình-môi-trường).

---

## Cấu trúc thư mục

```
OptimAIze/
├── main.py                       # entry point Gradio UI parent
├── packages/parent-core/         # lõi: app_ui.py + modules/ocr_bridge.py (cầu nối)
├── apps/parent-api/              # FastAPI parent
├── apps/parent-web/              # React + Vite parent
└── modules/<tên module>/         # các module con độc lập (git submodule)
    └── OptimAIze-OCR/            # module OCR
```

Xem [ARCHITECTURE.md](ARCHITECTURE.md) để hiểu cơ chế bridge parent ↔ module con và cách thêm module mới.
