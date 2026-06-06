# Kiến trúc OptimAIze

Tài liệu này tổng hợp kiến trúc, luồng xử lý và cách vận hành của dự án OptimAIze (parent workspace + submodule OptimAIze-OCR).

> Tài liệu mô tả mã nguồn tại thời điểm phân tích. Khi code đổi, hãy cập nhật lại các tham chiếu `file:line`.

---

## 1. Tổng quan

OptimAIze là một **monorepo dạng parent–child module**:

- **Parent workspace** (`OptimAIze/`) — lớp điều phối (orchestrator). Không tự chứa engine AI; chỉ phát hiện, điều phối và bắc cầu tới các module con.
- **Child module** (`modules/OptimAIze-OCR/`, git submodule) — engine OCR layout-aware tối ưu cho CPU, chạy được hoàn toàn độc lập.

Triết lý: **module độc lập, kích hoạt tường minh**. Mỗi module con có source / UI / API / database riêng. Parent chỉ resolve đường dẫn rồi gọi qua một lớp cầu nối mỏng (bridge), không nhúng cứng nội bộ của con. Model nặng chỉ load **sau hành động tường minh** của người dùng (lazy loading).

---

## 2. Cấu trúc thư mục

```
OptimAIze/  (PARENT)
├── main.py                              # entry point Gradio UI parent (port 7850)
├── pyproject.toml                       # deps nhẹ: gradio, pillow, pytest
├── packages/parent-core/optimaize/
│   ├── app_ui.py                        # Gradio UI parent
│   └── modules/ocr_bridge.py            # CẦU NỐI parent → OCR
├── apps/parent-api/                     # FastAPI parent (port 8000)
│   └── optimaize_parent_api/
│       ├── main.py
│       ├── api/v1/routes_modules.py     # liệt kê module, launch UI con
│       └── services/modules/            # module_registry, ocr_module_service
├── apps/parent-web/                     # React + Vite (port 5174)
│   └── src/
│       ├── app/App.tsx
│       ├── features/{overview,modules}/
│       └── shared/api/{client,parentApi}.ts
├── tests/parent/test_ocr_bridge.py
│
└── modules/OptimAIze-OCR/  (CHILD — submodule)
    ├── main.py                          # CLI OCR
    ├── pyproject.toml                   # deps nặng: torch, transformers, numba...
    ├── docker-compose.yml               # api (8001) + web (8080)
    ├── src/optimaize_ocr/               # ENGINE OCR
    │   ├── core/pipeline.py             # LayoutAwareOCRPipeline
    │   ├── backends/                    # 7 backend VLM
    │   │   └── dots_mocr/custom_backend.py
    │   ├── compute/                     # tối ưu CPU (AVX2, INT8, fused kernels)
    │   ├── output/{markdown,html}.py
    │   ├── prompts/category_prompts.py
    │   └── storage/history_db.py        # SQLite
    ├── apps/api/                        # FastAPI OCR (port 8001)
    ├── apps/web/                        # React + Vite (port 5173)
    └── legacy/gradio/app_ui.py          # UI Gradio cũ (port 7860) — parent gọi tới
```

---

## 3. Tầng tích hợp parent → child (`ocr_bridge.py`)

`packages/parent-core/optimaize/modules/ocr_bridge.py` là điểm tiếp xúc duy nhất giữa parent và OCR. Có 2 cơ chế:

1. **Bridge call trực tiếp** — `run_single_image_ocr()` dùng context manager `child_import_path()` chèn tạm `sys.path`, `import optimaize_ocr` rồi gọi pipeline ngay trong process parent. Lazy import: model chỉ load sau khi người dùng bấm chạy.
2. **Launch child UI** — `launch_child_ui()` dùng `subprocess.Popen` khởi chạy UI Gradio của con như tiến trình độc lập (port 7860), giữ tính tách biệt.

`child_status()` kiểm tra sự tồn tại của các file con (source / UI / history) và trả về trạng thái cho parent hiển thị.

> ⚠️ Điểm dễ vỡ: `PROJECT_ROOT = Path(__file__).resolve().parents[4]` (`ocr_bridge.py:14`) định vị submodule theo độ sâu thư mục cố định. Nếu di chuyển file, phải đổi con số này.

---

## 4. Engine OCR — luồng xử lý (`core/pipeline.py`)

`LayoutAwareOCRPipeline.parse()` chạy end-to-end 5 bước:

1. **Load ảnh** → convert RGB (`pipeline.py:135`)
2. **Layout Detection** — chạy `PP-DocLayoutV3` phát hiện cấu trúc (đoạn văn, tiêu đề, bảng, công thức) (`pipeline.py:196`)
3. **Filter & Crop** — cắt riêng từng vùng có chữ; bỏ qua hình/chart và vùng < 8px (`pipeline.py:201`)
4. **VLM OCR per crop** — mỗi crop qua một backend Vision-Language Model, dùng prompt theo đúng category (`pipeline.py:260`)
5. **Assemble Markdown** — ghép thành document có cấu trúc (`pipeline.py:297`)

Hai nhánh đặc biệt:
- **Fallback full-page** — khi không có vùng layout nào, OCR toàn trang (`pipeline.py:233`).
- **skip_layout** — cho dots-mocr tự sinh bbox ở chế độ full-page LAYOUT/SVG (`pipeline.py:156`).

Kết quả mỗi vùng: `{"category", "bbox": [x1,y1,x2,y2], "score", "text"}`. `pipeline.last_timings` lưu breakdown thời gian từng bước.

### Backend VLM

`backends/__init__.py:get_vlm_backend()` là factory chọn backend theo tên model:

| Model | Mặc định INT8 | Ghi chú |
|---|---|---|
| `falcon-ocr` | False | 400M, pure-PyTorch CPU, nhanh nhất load |
| `lighton-ocr` | True | 2.1B, tỉ lệ tốc độ/chất lượng tốt nhất |
| `dots-mocr` | True | ~3B, layout-aware, dùng `OptimizedDotsMOCRBackend` |
| `paddleocr-vl`, `surya-ocr`, `surya-package`, `glm-ocr` | — | các backend bổ sung |

Mọi backend kế thừa `BaseVLMBackend` (`backends/base.py`) với hợp đồng `generate_ocr(image, category) -> str`.

> Link model HuggingFace cho từng backend + layout detector, cùng các bộ benchmark tham khảo: xem [BENCHMARKS.md](BENCHMARKS.md).

---

## 5. Engine tối ưu CPU (`compute/`)

Mục tiêu: chạy VLM ~3B params trên CPU laptop với latency chấp nhận được (dots-mocr: ~170s → ~4.88s/crop).

### Quan sát gốc: GEMV vs GEMM

Khi sinh text tự hồi quy (decode), mỗi bước chỉ xử lý 1 token → input `(1,1,D)`. Phép `(1,1,D) @ W.T` là **GEMV** (matrix-vector), không phải GEMM. BLAS của PyTorch tối ưu mạnh cho GEMM nhưng kém cho GEMV "tall-skinny" ở batch=1. Toàn bộ engine xoay quanh việc thay `nn.Linear.forward` ở decode bằng kernel GEMV chuyên dụng.

### Kiến trúc 2 đường: prefill vs decode

Mọi `Linear` được vá để phân nhánh theo shape input:
- **Prefill** (seq dài): đường BLAS GEMM FP32 — vốn đã nhanh.
- **Decode** (`(1,1,D)`/`(1,D)`): kernel GEMV AVX2/Numba (`linear_dispatch.py:23`, `int8_linear.py:183`).

### Stack tối ưu 6 lớp (lắp ráp trong `custom_backend.py:546-574`)

| Lớp | File | Cách làm |
|---|---|---|
| 1. INT8 layer-by-layer | `quantization.py` | Quantize từng decoder layer rồi `gc.collect()` ngay — tránh đỉnh RAM ~8GB |
| 2. Patch Linear → GEMV | `linear_dispatch.py` | Monkey-patch `forward`, cache weight contiguous + output buffer sẵn |
| 3. Fused QKV | `fused_mlp.py:246` | Gộp q/k/v_proj thành 1 kernel đọc `x` một lần; cache K,V qua sentinel `(id(x),shape)` |
| 4. Folded RMSNorm+QKV | `fused_mlp.py:362` | Biến `input_layernorm` thành identity, gập norm vào kernel QKV |
| 5. Fused SwiGLU MLP | `fused_mlp.py:63` | Gộp gate+up+silu+nhân thành 1 pass, không vật chất hóa tensor trung gian |
| 6. INT8 lm_head | `int8_linear.py` | lm_head (1536→~152k vocab) — bottleneck đơn lớn nhất (~25ms → ~6.5ms) |

### Fallback 3 tầng (`compute/avx2/avx2_backend.py`)

Lúc import, hệ tự compile `avx2_gemv.cpp` (thử g++ → clang++ → MSVC cl.exe). Nếu không có compiler C++:

```
AVX2 C++ (FMA + OpenMP)  →  Numba JIT (LLVM + prange)  →  pure numpy
```

- Tự rebuild khi `.cpp` mới hơn `.dll` (so mtime, `_dll_is_stale()`).
- Kernel `m4` blocked khi `out_features % 4 == 0` để tái dùng lần load `x`.
- Chỉ `gemv_int8_avx2` / `gemv_float32_avx2` là bắt buộc; kernel fused là optional — thiếu thì tắt riêng lớp đó.

### Chi tiết kỹ thuật

- **Per-channel symmetric INT8** (`int8_linear.py:32`): mỗi output channel có scale `abs_max/127`, zero-point=0.
- **Double-buffer output** (`int8_linear.py:128`): luân phiên 2 buffer numpy, tránh copy mà vẫn ổn định tensor trả về.
- **`selective` vs `full` quantize**: dots.mocr **hallucinate/lặp vô hạn** nếu INT8 hóa attention → mặc định `selective` (INT8 MLP, attention FP32, ~6GB). `full` (~4GB nhưng hỏng output). `fp16` (~5GB, cân bằng). `none` (FP32, ~16GB).
- **Mock flash_attn** (`custom_backend.py:17`) + ép `attn_implementation="eager"` vì CPU không có flash-attention.
- **Cache quantized weights** ra `.pt` riêng theo mode — lần sau load thẳng.

---

## 6. Luồng API + Frontend

Hai cặp (API + Web) tách biệt. Frontend đều React 19 + Vite + TS, gọi REST qua `fetch` thuần, chỉ dùng `useState/useEffect/useMemo` (không Redux/axios/React-Query).

| | Parent | OCR |
|---|---|---|
| API port | 8000 | 8001 |
| Web port | 5174 | 5173 |
| API base (FE) | hard-code `:8000` | `VITE_OCR_API_BASE_URL` ?? `:8001` |
| Routing | anchor `#overview/#modules` | hash router (`intro/workspace/history`) |
| Vai trò API | liệt kê module, launch UI con | OCR thật + history |

### Luồng Parent — điều phối

```
parent-web/App.tsx
  → getModules()                    [GET :8000/api/v1/modules]
  → list_modules() → ocr_status() → ocr_bridge.child_status()
  ← ModuleStatus (available, paths, web_url, api_url)

[Launch] → launchOcrLegacyUi()      [POST .../ocr/launch-ui]
  → ocr_bridge.launch_child_ui()    [subprocess.Popen UI Gradio]
```

### Luồng OCR — xử lý ảnh (cốt lõi)

```
WorkspacePage → useSingleImageOCR (hook giữ toàn bộ state)
  submit() → runSingleImageOCR({image, modelType, config})
    → FormData multipart           [POST :8001/api/v1/ocr/single-image]
      ↓
routes_ocr.single_image_ocr()
  → FileStore.save_upload()        [validate type/size, convert RGB, lưu .png]
  → OCRService.run_single_image()
    → get_pipeline_cached()        [lru_cache(1) — giữ 1 model trong RAM]
    → pipeline.parse()             [5 bước: layout→crop→VLM→markdown]
    → pipeline.generate_html()
    → ghi parsed_document.md + .html ra outputs/api_runs/<ts>/
  ← {markdown, html, regions[], timings, output_dir, image_name}
```

### Chi tiết frontend

- **Config phụ thuộc model** (`useSingleImageOCR.ts:81`): non-`dots-mocr` bị ép `skipLayout=false, fullPageMode='layout'`.
- **`quantizeInt8='auto'`** không gửi lên server → backend tự quyết default per-model.
- **Progress giả lập** (20%→65%→100%) vì API là request-response đơn, không streaming.
- **Preview blob** quản lý đúng bằng `URL.createObjectURL` + `revokeObjectURL` trong cleanup.

### Lưu trữ & history

- **FileStore** (`file_store.py`): chỉ nhận png/jpeg/webp/bmp/tiff, ≤25MB, convert RGB, `safe_name` chống path traversal.
- **History** SQLite (`storage/history_db.py`, `history_repository.py`): REST cho documents/runs/export-text/delete.
- Output ghi ra `outputs/api_runs/<timestamp_ms>/`.

---

## 7. Cách chạy

### Yêu cầu môi trường

- **Python 3.12** (khuyến nghị — stack OCR pin `torch==2.9.1`, `transformers==5.4.0`; Python 3.14 có thể thiếu wheel).
- **Node.js** (cho frontend React/Vite).
- (Tùy chọn) compiler C++ với AVX2 (g++/clang++/MSVC) để bật kernel AVX2; thiếu thì tự fallback Numba.

### Cài đặt

```bash
# 1. Tạo venv Python 3.12
py -3.12 -m venv .venv

# 2. Cài parent (nhẹ — chạy được UI ngay cả khi chưa cài OCR)
.venv/Scripts/python.exe -m pip install -e .

# 3. Cài OCR submodule (nặng — torch, transformers... vài GB)
.venv/Scripts/python.exe -m pip install -e modules/OptimAIze-OCR

# 4. Cài frontend
cd apps/parent-web && npm install
cd modules/OptimAIze-OCR/apps/web && npm install
```

### Chạy

```bash
# Parent Gradio UI (port 7850)
python main.py

# Parent API (port 8000)
uvicorn optimaize_parent_api.main:app --app-dir apps/parent-api --port 8000

# Parent web (port 5174)
cd apps/parent-web && npm run dev

# --- OCR (độc lập) ---
# OCR API (port 8001)
uvicorn optimaize_ocr_api.main:app --app-dir modules/OptimAIze-OCR/apps/api --port 8001

# OCR web (port 5173)
cd modules/OptimAIze-OCR/apps/web && npm run dev

# OCR UI Gradio cũ (port 7860)
python modules/OptimAIze-OCR/legacy/gradio/app_ui.py --server-port 7860

# OCR qua Docker (api 8001 + web 8080)
cd modules/OptimAIze-OCR && docker compose up
```

### Test

```bash
.venv/Scripts/python.exe -m pytest tests/ modules/OptimAIze-OCR/apps/api/optimaize_ocr_api/tests/ -v
```

---

## 8. Lưu ý vận hành & bảo mật

- **Không có authentication**: cả parent-api lẫn ocr-api mở `/api/v1` không kiểm soát truy cập; CORS rộng (`allow_credentials=True`). Phù hợp chạy local; **không nên expose ra mạng** nếu chưa thêm auth.
- **`lru_cache(maxsize=1)`** trên pipeline: đổi bất kỳ tham số runtime nào → cache miss → reload model (hàng chục giây). Chủ đích: chỉ giữ 1 model trong RAM.
- **Dependencies pinned cao** (transformers 5.4.0, gradio 6.1.0): tốt cho reproducibility, có thể khó cài trên môi trường cũ.
- **Cài OCR downgrade một số package parent** (gradio 6.16→6.1, fastapi 0.136→0.124) theo pin của submodule — dùng chung một venv nên lấy theo ràng buộc chặt hơn.
- **API base hard-code `:8000`** ở parent nav (`App.tsx:47`) — kém linh hoạt hơn OCR app (dùng env var).

---

## 9. Cấu hình môi trường (OCR API)

`apps/api/optimaize_ocr_api/core/config.py` đọc các biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OPTIMAIZE_OCR_OUTPUT_DIR` | `DEFAULT_OUTPUT_DIR` | thư mục output |
| `OPTIMAIZE_OCR_UPLOAD_DIR` | `DEFAULT_UPLOAD_DIR` | thư mục upload |
| `OPTIMAIZE_OCR_HISTORY_DB` | `DEFAULT_HISTORY_DB` | đường dẫn SQLite history |
| `OPTIMAIZE_OCR_CORS_ORIGINS` | danh sách localhost | CORS origins (CSV) |
| `OPTIMAIZE_OCR_MAX_UPLOAD_BYTES` | 25MB | giới hạn upload |

Docker (`docker-compose.yml`) còn set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`... = 4 và giới hạn 3.75 CPU / 3840MB RAM cho service api.
