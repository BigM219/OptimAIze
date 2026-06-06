# Kiến trúc OptimAIze (Parent)

Tài liệu này mô tả **lớp điều phối parent**. Mọi chi tiết về một module con nằm trong chính module đó — ví dụ OCR: [modules/OptimAIze-OCR/ARCHITECTURE.md](modules/OptimAIze-OCR/ARCHITECTURE.md).

---

## 1. Triết lý: module độc lập, kích hoạt tường minh

- Mỗi **module con** chạy được hoàn toàn độc lập — có source, UI, API, database riêng — và gắn vào parent qua git submodule.
- **Parent** chỉ là lớp điều phối mỏng: phát hiện module, hiển thị trạng thái, bắc cầu (bridge) sang module con. Parent không nhúng cứng nội bộ của con.
- **Lazy loading**: parent kiểm tra trạng thái lúc khởi động, nhưng model/tài nguyên nặng của con chỉ load sau hành động tường minh của người dùng.

Hệ quả: thêm module mới = thêm một submodule + một bridge, không làm phình lõi parent.

---

## 2. Thành phần parent

```
OptimAIze/
├── main.py                                  # entry point Gradio UI parent (7850)
├── packages/parent-core/optimaize/
│   ├── app_ui.py                            # Gradio UI parent
│   └── modules/ocr_bridge.py                # CẦU NỐI parent → OptimAIze-OCR
├── apps/parent-api/                         # FastAPI parent (8000)
│   └── optimaize_parent_api/
│       ├── main.py
│       ├── core/{config,errors,auth}.py
│       ├── api/v1/{routes_health,routes_modules}.py
│       └── services/modules/                # module_registry, ocr_module_service
└── apps/parent-web/                         # React + Vite (5174)
```

---

## 3. Cơ chế bridge (parent → module con)

`packages/parent-core/optimaize/modules/ocr_bridge.py` là điểm tiếp xúc duy nhất giữa parent và OCR. Hai cơ chế:

1. **Bridge call trực tiếp** — context manager `child_import_path()` chèn tạm `sys.path`, `import` package của con rồi gọi trong process parent. Lazy import: model chỉ load khi cần.
2. **Launch child UI** — `subprocess.Popen` khởi chạy UI của con như tiến trình độc lập (giữ tính tách biệt).

`child_status()` kiểm tra các file của con tồn tại và trả về trạng thái cho parent hiển thị.

> Định vị workspace root: `_discover_project_root()` ưu tiên env `OPTIMAIZE_PROJECT_ROOT`, rồi walk-up tìm thư mục chứa `modules/`, cuối cùng mới fallback fixed-depth.

---

## 4. Parent API

`apps/parent-api/`. Router dưới `/api/v1`:

| Method | Path | Loại |
|---|---|---|
| GET | `/health` | mở (probe) |
| GET | `/api/v1/modules` | đọc — liệt kê module |
| GET | `/api/v1/modules/ocr/status` | đọc |
| POST | `/api/v1/modules/ocr/launch-ui` | đổi trạng thái (spawn UI con) |

Luồng liệt kê module:

```
parent-web/App.tsx → getModules()        [GET :8000/api/v1/modules]
  → list_modules() → ocr_status()
  → ocr_bridge.child_status()            [kiểm tra file submodule]
  ← ModuleStatus (available, paths, web_url, api_url)
```

### Bảo mật

- `core/auth.py`: dependency `require_api_key` — bật khi `OPTIMAIZE_API_KEY` được set (client gửi `X-API-Key`); không set → auth tắt. Áp cho router `modules`; `/health` mở.
- CORS qua `OPTIMAIZE_PARENT_CORS_ORIGINS` (CSV).

### Frontend

`apps/parent-web/` — React 19 + Vite + fetch thuần. API base qua `VITE_PARENT_API_URL` (mặc định `:8000`); link health dùng `HEALTH_URL` dẫn xuất từ env, không hard-code.

---

## 5. Thêm module mới

1. Thêm submodule vào `modules/<tên>/`.
2. Viết một bridge trong `packages/parent-core/optimaize/modules/` (theo mẫu `ocr_bridge.py`).
3. Đăng ký vào `services/modules/module_registry.py` để parent liệt kê được.
4. Tài liệu chi tiết của module → đặt **trong module đó**, parent chỉ dẫn sang.

---

## 6. Lưu ý vận hành

- **Không auth khi `OPTIMAIZE_API_KEY` chưa set** + CORS rộng mặc định — phù hợp local, không nên expose ra mạng nếu chưa siết.
- Dùng chung một venv với module con: ràng buộc version chặt hơn (của con) sẽ thắng. Cài parent trước (nhẹ) rồi tới module con.
