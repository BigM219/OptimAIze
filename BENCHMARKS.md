# Model Links & Benchmark References

Tài liệu này gom (1) link model HuggingFace cho mọi backend OptimAIze-OCR dùng, và (2) các bộ benchmark tham khảo để đánh giá chất lượng trích xuất tài liệu.

---

## 1. Model links

### VLM OCR backends

Các model do `get_vlm_backend()` ([backends/__init__.py](modules/OptimAIze-OCR/src/optimaize_ocr/backends/__init__.py)) nạp. Tên model (cột "Tên gọi") là giá trị truyền vào `model_type`.

| Tên gọi | HuggingFace repo | Kích thước | Ghi chú |
|---|---|---|---|
| `falcon-ocr` | [tiiuae/Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) | ~400M | Pure-PyTorch CPU, nhanh nhất load |
| `lighton-ocr` | [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | ~2.1B | Tỉ lệ tốc độ/chất lượng tốt nhất |
| `dots-mocr` | [rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) | ~3B | Layout-aware, dùng OptimizedDotsMOCRBackend |
| `paddleocr-vl` | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | — | Cũng hỗ trợ biến thể `PaddleOCR-VL-1.5` |
| `glm-ocr` | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | — | |
| `surya-ocr` | [datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2) | — | |
| `surya-package` | (gói pip `surya-ocr`) | — | Dùng package Surya thay vì repo HF trực tiếp |

### Layout detector

| Vai trò | HuggingFace repo | Ghi chú |
|---|---|---|
| Phát hiện cấu trúc tài liệu | [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) | Chạy trước mọi VLM; phát hiện đoạn văn/tiêu đề/bảng/công thức |

> Model được tải tự động qua `huggingface_hub` lần đầu chạy và cache lại. Weights quantized của dots-mocr được lưu thêm ra `weights/` dưới dạng `.pt` (xem `OPTIMAIZE_OCR_WEIGHTS_DIR`).

---

## 2. Benchmark references

### opendataloader-pdf — extraction benchmarks

Nguồn chính bạn yêu cầu: [opendataloader-pdf #extraction-benchmarks](https://github.com/opendataloader-project/opendataloader-pdf#extraction-benchmarks) · repo benchmark đầy đủ: [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench).

- **Bộ test**: 200 PDF thực tế (gồm tài liệu nhiều cột và bài báo khoa học).
- **Chuẩn hóa điểm** về [0, 1] — cao hơn = tốt hơn (riêng Speed s/page thì thấp hơn = tốt hơn).
- **Đo trên** Apple M4, không cần GPU.
- **Metrics**: Overall · Reading Order · Table · Heading · Speed (s/page) · License.

Bảng kết quả (trích từ README opendataloader-pdf, để tham chiếu so sánh):

| Engine | Overall | Reading Order | Table | Heading | Speed (s/page) | License |
|---|---|---|---|---|---|---|
| opendataloader [hybrid] | 0.907 | 0.934 | 0.928 | 0.821 | 0.463 | Apache-2.0 |
| nutrient | 0.885 | 0.925 | 0.708 | 0.819 | 0.008 | Commercial |
| docling | 0.882 | 0.898 | 0.887 | 0.824 | 0.762 | MIT |
| marker | 0.861 | 0.890 | 0.808 | 0.796 | 53.932 | GPL-3.0 |
| unstructured [hi_res] | 0.841 | 0.904 | 0.588 | 0.749 | 3.008 | Apache-2.0 |
| edgeparse | 0.837 | 0.894 | 0.717 | 0.706 | 0.036 | Apache-2.0 |
| opendataloader | 0.831 | 0.902 | 0.489 | 0.739 | 0.015 | Apache-2.0 |
| mineru | 0.831 | 0.857 | 0.873 | 0.743 | 5.962 | AGPL-3.0 |
| pymupdf4llm | 0.732 | 0.885 | 0.401 | 0.412 | 0.091 | AGPL-3.0 |
| unstructured | 0.686 | 0.882 | 0.000 | 0.388 | 0.077 | Apache-2.0 |
| markitdown | 0.589 | 0.844 | 0.273 | 0.000 | 0.114 | MIT |
| liteparse | 0.576 | 0.866 | 0.000 | 0.000 | 1.061 | Apache-2.0 |

> Lưu ý license: bảng gồm nhiều engine với license khác nhau (MIT, GPL, AGPL, Commercial). Tôn trọng license riêng của từng công cụ nếu dùng.

### Các bộ benchmark OCR/document-parsing tham khảo thêm

| Benchmark | Link | Tập trung vào |
|---|---|---|
| OmniDocBench | [github.com/opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Đánh giá đa dạng loại tài liệu (text/table/formula/reading-order) |
| olmOCR-bench | [github.com/allenai/olmocr](https://github.com/allenai/olmocr) | Benchmark OCR cho pipeline olmOCR của AllenAI |
| Marker | [github.com/datalab-to/marker](https://github.com/datalab-to/marker) | PDF→Markdown, có bộ so sánh riêng |
| Docling | [github.com/docling-project/docling](https://github.com/docling-project/docling) | Trích xuất tài liệu của IBM |

---

## 3. Benchmark nội bộ của OptimAIze-OCR

Dự án đã có sẵn công cụ benchmark per-backend (đo latency/crop, không cần GPU):

- `tools/benchmarks/benchmark_ocr_dataset.py` — chạy backend trên một dataset ảnh.
- `tools/benchmarks/benchmark_multipage_docqa.py` — benchmark DocQA nhiều trang.
- `tools/benchmarks/sweep_threads.py` — quét số luồng tìm cấu hình tối ưu.
- `tools/profiling/` — profiler per-Linear / per-module cho vòng decode.

Tham khảo bảng hiệu năng per-crop và RAM trong [README của submodule](modules/OptimAIze-OCR/README.md#performance).

> ⚠️ Khi chạy benchmark trên máy này (6 nhân vật lý / 15.36GB), lớp resource governance ([resource_policy.py](modules/OptimAIze-OCR/src/optimaize_ocr/resource_policy.py)) sẽ giới hạn luồng theo `OPTIMAIZE_ENV` (dev=50% / prod=80%) và từ chối model vượt trần RAM. Đặt `OPTIMAIZE_ENV=prod` để benchmark sát ngưỡng cao hơn.
