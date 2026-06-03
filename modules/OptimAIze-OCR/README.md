# Flexible CPU-Optimized Layout-Aware OCR Pipeline

A modular, highly optimized Python pipeline for Layout-Aware OCR designed to run efficiently on **CPUs** — no GPU required.

The pipeline runs `PP-DocLayoutV3` to detect document structures (paragraphs, titles, tables, formulas), crops each region, then feeds it to a selected Vision-Language Model (VLM) backend. Each backend is independently optimized: INT8 quantization, fused AVX2/OpenMP kernels, and kernel fusion keep RAM and latency low on standard laptop/server hardware.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models Overview](#models-overview)
  - [Falcon-OCR](#1-falcon-ocr-tiiuaefalcon-ocr)
  - [LightOn-OCR](#2-lighton-ocr-lightonaililightonocr-2-1b)
  - [Dots-MOCR (Standard)](#3-dots-mocr-standard-rednotehilabdotsmocr)
  - [Dots-MOCR (Optimized)](#4-dots-mocr-optimized--recommended-for-cpu)
- [Python API](#python-api)
  - [LayoutAwareOCRPipeline](#layoutawareocrpipeline)
  - [get_vlm_backend (low-level)](#get_vlm_backend-low-level)
- [CLI Usage](#cli-usage)
  - [main.py — Full Pipeline](#mainpy--full-pipeline)
  - [scripts/bench.py — Backend Benchmark](#scriptsbenchpy--backend-benchmark)
  - [scripts/sweep_threads.py — Thread Tuning](#scriptssweep_threadspy--thread-tuning)
- [Performance](#performance)
- [Thread Tuning Guide](#thread-tuning-guide)
- [Advanced: Backend-level API](#advanced-backend-level-api)

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd flexible-cpu-ocr
pip install -e .
```

**Dependencies:** PyTorch (CPU), Transformers, Pillow, NumPy, Numba, tqdm.  
**Optional (for AVX2 kernels):** `g++` with AVX2 support (auto-compiled on first use).

---

## Quick Start

```python
from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline, setup_cpu_optimization

# Tune thread count before loading (set once)
setup_cpu_optimization(num_threads=8)

# Initialize pipeline
pipeline = LayoutAwareOCRPipeline(
    model_type="dots-mocr",       # "falcon-ocr" | "lighton-ocr" | "dots-mocr"
    quantize_int8=True,           # INT8 quantization — cuts RAM ~4x
    num_threads=8,                # Physical cores (not hyperthreads)
)

# Run OCR
markdown, results = pipeline.parse("path/to/document.png")
print(markdown)

# Each result is a dict: {"category": "text", "bbox": [x1,y1,x2,y2], "score": 0.95, "text": "..."}
for r in results:
    print(f"[{r['category']}] {r['text'][:80]}")
```

---

## Models Overview

| Model | Size | `quantize_int8` default | Recommended threads | Small crop | Medium crop | Notes |
|---|---|---|---|---|---|---|
| `falcon-ocr` | 400M | `False` | 4–8 | ~1s | ~5.5s | Fastest load, pure PyTorch |
| `lighton-ocr` | 2.1B | `True` | 4–8 | ~0.66s | ~1.38s | Best speed/quality ratio |
| `dots-mocr` (standard) | ~3B | `True` | 8 | ~170s | — | Layout-aware, slow without fusion |
| `dots-mocr` + optimized | ~3B | `True` | 8 | **~4.88s** | ~15s | Fused kernels — **recommended** |

---

## 1. Falcon-OCR (`tiiuae/Falcon-OCR`)

A compact 400M-parameter model. Uses a custom pure-PyTorch CPU engine that bypasses CUDA/Triton (`FlexAttention`) restrictions while remaining 100% mathematically equivalent.

**Best for:** Fast document OCR where speed and low RAM are prioritized.

```python
from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline

pipeline = LayoutAwareOCRPipeline(
    model_type="falcon-ocr",
    quantize_int8=False,   # Default: False (small model, quantization hurts fidelity)
    num_threads=8,
)

markdown, results = pipeline.parse("document.png")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_type` | — | `"falcon-ocr"` |
| `quantize_int8` | `False` | Typically not needed — model is small (400M) |
| `num_threads` | `None` (all) | Recommended: 4–8 physical cores |
| `layout_threshold` | `0.3` | PP-DocLayoutV3 confidence threshold |

---

## 2. LightOn-OCR (`lightonai/LightOnOCR-2-1B`)

A 2.1B-parameter model running via standard HuggingFace `transformers`. Provides the best speed/quality trade-off. INT8 quantization enabled by default to keep RAM under control.

**Best for:** Production-grade OCR workloads where accuracy matters and latency should stay under 2s per crop.

```python
from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline

pipeline = LayoutAwareOCRPipeline(
    model_type="lighton-ocr",
    quantize_int8=True,    # Default: True (cuts 2.1B → ~600MB RAM)
    num_threads=8,
)

markdown, results = pipeline.parse("document.png")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_type` | — | `"lighton-ocr"` |
| `quantize_int8` | `True` | Strongly recommended — 2.1B without INT8 uses ~8GB RAM |
| `num_threads` | `None` (all) | Recommended: 4–8 physical cores |
| `layout_threshold` | `0.3` | PP-DocLayoutV3 confidence threshold |

---

## 3. Dots-MOCR (Standard) (`rednote-hilab/dots.mocr`)

The full ~3B-parameter multimodal model with layout-aware prompting. Uses vendor-specific prompt format (`DOTS_MOCR_CATEGORY_PROMPTS`) for each layout category. Standard mode uses the default HuggingFace `generate()` loop — correct but slow on CPU (~170s/crop without fusion).

**Use this only for debugging.** For actual use, always prefer the optimized variant.

```python
from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline

pipeline = LayoutAwareOCRPipeline(
    model_type="dots-mocr",
    quantize_int8=True,
    num_threads=8,
)
```

---

## 4. Dots-MOCR (Optimized) — Recommended for CPU

Same model with all CPU optimizations stacked:

1. **FBGEMM INT8 quantization** — layer-by-layer dynamic INT8 for decoder
2. **Per-channel INT8 `lm_head`** — reduces the largest single bottleneck from ~25ms → ~6.5ms/call
3. **AVX2+OpenMP GEMV kernels** — custom C++ kernels for all `nn.Linear` decode paths
4. **Fused SwiGLU MLP** — fuses `gate_proj`, `up_proj`, SiLU, and `down_proj` into one kernel
5. **Fused QKV projection** — fuses `q_proj`, `k_proj`, `v_proj` fan-out into one kernel
6. **Folded RMSNorm + QKV** — pre-folds `input_layernorm` weights into the QKV kernel (eliminates the normalization step at decode time)

**Best for:** High-quality layout-aware OCR at acceptable CPU latency (~5s for a small text crop).

```python
from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import setup_cpu_optimization

setup_cpu_optimization(num_threads=8)

backend = get_vlm_backend(
    model_type="dots-mocr",
    device="cpu",
    quantize_int8=True,          # Default: True
    use_optimized_dots=True,     # REQUIRED to enable all fused kernels
)

from PIL import Image
img = Image.open("document_crop.png").convert("RGB")
text = backend.generate_ocr(img, category="text")
print(text)
```

Or via the unified pipeline:

```python
# LayoutAwareOCRPipeline does NOT expose use_optimized_dots directly.
# Use get_vlm_backend() and call generate_ocr() per crop, or use
# the pipeline with the direct backend injection (see Advanced section).
```

**Parameters for `get_vlm_backend`:**

| Parameter | Default | Description |
|---|---|---|
| `model_type` | — | `"dots-mocr"` |
| `device` | `"cpu"` | Only CPU is supported |
| `quantize_int8` | `True` | INT8 quantization for decoder layers |
| `use_optimized_dots` | `False` | **Set `True`** to enable fused kernels |

**`OptimizedDotsMOCRBackend` internal settings:**

| Setting | Value | Description |
|---|---|---|
| `max_new_tokens` | `256` | Max decode length per crop |
| `max_vision_tokens` | `256` | Vision token budget |
| `attn_implementation` | `"eager"` | Required — DotsVisionTransformer doesn't support sdpa |

---

## Python API

### `LayoutAwareOCRPipeline`

```python
LayoutAwareOCRPipeline(
    model_type: str = "falcon-ocr",
    layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    num_threads: int | None = None,
    device: str = "cpu",
    quantize_int8: bool | None = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `model_type` | `"falcon-ocr"` | VLM backend: `"falcon-ocr"` \| `"lighton-ocr"` \| `"dots-mocr"` |
| `layout_model` | PP-DocLayoutV3 | HuggingFace model ID for layout detector |
| `num_threads` | `None` | Thread count for PyTorch, OpenMP, MKL, Numba. `None` = use OS default |
| `device` | `"cpu"` | Inference device. Only CPU is supported. |
| `quantize_int8` | `None` | `None` uses per-model default: `False` for Falcon-OCR, `True` for others |

**`pipeline.parse(image_path, layout_threshold=0.3, save_crops_dir=None)`**

Runs the full end-to-end pipeline:

```python
markdown: str, results: list[dict] = pipeline.parse(
    image_path="document.png",
    layout_threshold=0.3,          # Detection confidence threshold
    save_crops_dir="outputs/crops" # Optional: saves table/formula crops as PNG
)
```

Returns:
- `markdown` — Assembled markdown string of the full document
- `results` — List of dicts, one per detected region:
  ```python
  {
      "category": "text",          # layout category: text | title | table | formula | ...
      "bbox": [x1, y1, x2, y2],   # pixel coordinates in original image
      "score": 0.95,               # detection confidence
      "text": "Recognized text..."
  }
  ```

**`pipeline.last_timings`** — Timing breakdown from last `parse()` call:

```python
{
    "overall_time": 6.23,
    "img_prep_time": 0.01,
    "layout_time": 0.88,
    "crop_time": 0.002,
    "ocr_time_total": 5.22,
    "markdown_time": 0.001
}
```

---

### `get_vlm_backend` (low-level)

```python
from optimaize_ocr.backends import get_vlm_backend

backend = get_vlm_backend(
    model_type: str,               # "falcon-ocr" | "lighton-ocr" | "dots-mocr"
    device: str = "cpu",
    quantize_int8: bool | None = None,
    use_optimized_dots: bool = False,  # Only applies to dots-mocr
) -> BaseVLMBackend
```

**`backend.generate_ocr(image: PIL.Image, category: str) -> str`**

Runs a single crop through the VLM. Category controls which prompt template is used:

```python
from PIL import Image

img = Image.open("crop.png").convert("RGB")
text = backend.generate_ocr(img, category="text")      # plain text
text = backend.generate_ocr(img, category="title")     # heading
text = backend.generate_ocr(img, category="table")     # structured table → markdown
text = backend.generate_ocr(img, category="formula")   # math formula → LaTeX
```

Valid categories: `"text"`, `"title"`, `"table"`, `"formula"`, `"plain"`.

---

### `setup_cpu_optimization`

```python
from optimaize_ocr.core.pipeline import setup_cpu_optimization

setup_cpu_optimization(num_threads=8)
```

Sets the thread count for **all** relevant thread pools at once:
- `torch.set_num_threads(n)`
- `OMP_NUM_THREADS=n` (OpenMP / AVX2 kernels)
- `MKL_NUM_THREADS=n`
- `numba.set_num_threads(n)`

Call once before loading the model. Changing `num_threads` after model load is supported but less reliable for BLAS.

---

## CLI Usage

### `main.py` — Full Pipeline

```bash
# Falcon-OCR (fastest)
python main.py --image path/to/document.png --model falcon-ocr

# LightOn-OCR (best speed/quality)
python main.py --image path/to/document.png --model lighton-ocr

# Dots-MOCR (highest quality, slow without --use-optimized-dots)
python main.py --image path/to/document.png --model dots-mocr

# Save markdown to output dir
python main.py --image document.png --model falcon-ocr --output-dir ./outputs/

# Control thread count
python main.py --image document.png --model lighton-ocr --threads 8

# Adjust layout detection threshold
python main.py --image document.png --model falcon-ocr --threshold 0.5
```

**CLI Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--image` | (required) | Path to input image |
| `--model` | `falcon-ocr` | `falcon-ocr` \| `lighton-ocr` \| `dots-mocr` |
| `--threshold` | `0.3` | Layout detection confidence threshold |
| `--threads` | all available | Number of CPU threads |
| `--output-dir` | `None` | Directory to write markdown + crop images |

---

### `scripts/bench.py` — Backend Benchmark

Benchmarks a backend directly on synthetic crops (bypasses the layout detector), measuring raw per-crop VLM latency.

```bash
# Benchmark Falcon-OCR
python scripts/bench.py --model falcon-ocr

# Benchmark LightOn-OCR with INT8 (default)
python scripts/bench.py --model lighton-ocr

# Benchmark Dots-MOCR without optimizations
python scripts/bench.py --model dots-mocr --quantize-int8

# Benchmark Dots-MOCR with all fused kernels (recommended)
python scripts/bench.py --model dots-mocr --quantize-int8 --use-optimized-dots --threads 8

# Disable INT8 explicitly
python scripts/bench.py --model lighton-ocr --no-quantize-int8

# Skip real document, test synthetic crops only
python scripts/bench.py --model dots-mocr --use-optimized-dots --synth-only
```

**Flags:**

| Flag | Description |
|---|---|
| `--model` | `falcon-ocr` \| `lighton-ocr` \| `dots-mocr` |
| `--threads N` | Thread count passed to `setup_cpu_optimization` |
| `--quantize-int8` | Force INT8 quantization ON |
| `--no-quantize-int8` | Force INT8 quantization OFF |
| `--use-optimized-dots` | Enable fused kernels for Dots-MOCR |
| `--synth-only` | Skip layout pipeline, benchmark only synthetic crops |

---

### `scripts/sweep_threads.py` — Thread Tuning

Sweeps a range of thread counts to find the optimal setting for your CPU.

```bash
# Default sweep: 2,4,6,8,10,12,14 threads, 2 repeats each
python scripts/sweep_threads.py

# Custom range and repeats
python scripts/sweep_threads.py --threads 4,6,8,10 --repeats 3

# Let EOS stop naturally (no token cap)
python scripts/sweep_threads.py --max-new-tokens 0

# Cap to 30 tokens for a faster sweep
python scripts/sweep_threads.py --max-new-tokens 30 --threads 4,6,8,10,12
```

Output example:
```
 threads |  avg(s) |  min(s) | output preview
------------------------------------------------------------------------
       4 |    7.21 |    7.18 | 'Page 12 — Footnote line of text.'
       6 |    5.93 |    5.88 | 'Page 12 — Footnote line of text.'
       8 |    4.88 |    4.85 | 'Page 12 — Footnote line of text.'
      10 |    5.12 |    5.09 | 'Page 12 — Footnote line of text.'
      12 |    5.44 |    5.41 | 'Page 12 — Footnote line of text.'
      14 |    5.83 |    5.79 | 'Page 12 — Footnote line of text.'

Fastest: 8 threads -> 4.85s
```

---

## Performance

All timings are on a 14-core Intel Meteor Lake CPU (E-core + P-core), 8 threads, INT8 enabled where applicable. Synthetic crops: `small` = 320×60px, `medium` = 640×200px, `large` = 800×400px.

### Per-crop latency (warm, after JIT compile)

| Model | small | medium | large |
|---|---|---|---|
| `falcon-ocr` (FP32) | ~1.0s | ~5.5s | ~12s |
| `lighton-ocr` (INT8) | ~0.66s | ~1.38s | ~3s |
| `dots-mocr` standard (INT8) | ~170s | — | — |
| `dots-mocr` optimized (INT8 + fused kernels) | **~4.88s** | **~15s** | ~45s |

### Model load time (first run, cold cache)

| Model | Load time |
|---|---|
| `falcon-ocr` | ~3s |
| `lighton-ocr` | ~8s |
| `dots-mocr` (optimized) | ~25s (incl. quantization + kernel fusion) |

### RAM usage (peak, with `quantize_int8=True`)

| Model | FP32 RAM | INT8 RAM |
|---|---|---|
| `falcon-ocr` | ~1.5GB | — |
| `lighton-ocr` | ~8GB | ~2GB |
| `dots-mocr` | ~12GB | ~3GB |

---

## Thread Tuning Guide

Thread count has a large impact on CPU inference latency. The optimal value depends on your CPU topology:

- **Physical cores only** (no hyperthreads): set `num_threads` = all physical cores.
- **Hyperthreaded CPUs**: start with physical core count. Hyperthreads share FMA units — too many threads wastes cycles on synchronization.
- **14-core (P+E) CPUs**: optimal is typically **8 threads** for Dots-MOCR; P-cores only.
- **8-core laptop CPUs**: try 4, 6, 8 — measure with `sweep_threads.py`.

The AVX2 kernels (SwiGLU, QKV, RMSNorm+QKV) use OpenMP. The `OMP_NUM_THREADS` env var must be set **before** importing PyTorch:

```bash
# Set before running
OMP_NUM_THREADS=8 python scripts/bench.py --model dots-mocr --use-optimized-dots
```

Or call `setup_cpu_optimization(8)` at the very start of your Python script.

---

## Advanced: Backend-level API

### Using `OptimizedDotsMOCRBackend` directly

```python
from optimaize_ocr.backends.dots_mocr.custom_backend import OptimizedDotsMOCRBackend
from optimaize_ocr.core.pipeline import setup_cpu_optimization

setup_cpu_optimization(num_threads=8)

backend = OptimizedDotsMOCRBackend(device="cpu", quantize_int8=True)

from PIL import Image
img = Image.open("crop.png").convert("RGB")

# category controls the vendor prompt used
text = backend.generate_ocr(img, category="text")
text = backend.generate_ocr(img, category="table")   # → markdown table
text = backend.generate_ocr(img, category="formula") # → LaTeX
```

### Optimization stack applied by `OptimizedDotsMOCRBackend`

Applied in this order at init time:

```
quantize_model_layer_by_layer()     # FBGEMM INT8 for decoder nn.Linear
patch_model_linear_layers()         # AVX2+OpenMP GEMV for all patched Linear
fuse_qwen2_attn_qkv()               # Fuse q/k/v_proj into single AVX2 kernel
fuse_qwen2_attn_rmsnorm()           # Fold input_layernorm into QKV kernel
fuse_qwen2_mlp_swiglu()             # Fuse gate_proj+up_proj+silu+down_proj
Int8Linear(lm_head)                 # Per-channel INT8 on lm_head (~25ms→6.5ms)
```

### Using `get_vlm_backend` with a custom pipeline

```python
from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import setup_cpu_optimization
from PIL import Image

setup_cpu_optimization(num_threads=8)
backend = get_vlm_backend("dots-mocr", quantize_int8=True, use_optimized_dots=True)

images = [Image.open(p).convert("RGB") for p in crop_paths]
texts = [backend.generate_ocr(img, category="text") for img in images]
```

### Category-to-prompt mapping

Dots-MOCR uses vendor-trained prompt templates. Wrong prompts cause hallucinated or garbage output. The mapping is in `optimaize_ocr/prompts/category_prompts.py`:

```python
from optimaize_ocr.prompts import DOTS_MOCR_CATEGORY_PROMPTS

# Shows the actual prompt string used for each category
for cat, prompt in DOTS_MOCR_CATEGORY_PROMPTS.items():
    print(f"{cat}: {prompt[:80]}")
```

---

## Project Structure

```
optimaize_ocr/
├── backends/
│   ├── falcon_ocr/         # Falcon-OCR custom pure-PyTorch CPU engine
│   ├── lighton_ocr/        # LightOn-OCR HuggingFace backend
│   └── dots_mocr/
│       ├── custom_backend.py   # OptimizedDotsMOCRBackend (fused kernels)
│       └── patches.py          # prepare_inputs_for_generation fix
├── compute/
│   ├── avx2/
│   │   └── avx2_gemv.cpp       # AVX2+OpenMP GEMV/GEMM/fused kernels
│   ├── fused_mlp.py            # SwiGLU, QKV, RMSNorm+QKV fusion
│   ├── int8_linear.py          # Per-channel INT8 Linear module
│   ├── quantization.py         # FBGEMM layer-by-layer quantization
│   └── linear_dispatch.py      # Dispatch to AVX2 / Numba / PyTorch
├── core/
│   └── pipeline.py             # LayoutAwareOCRPipeline + setup_cpu_optimization
└── prompts/
    └── category_prompts.py     # Per-model prompt templates

scripts/
├── bench.py                    # Backend benchmark
├── sweep_threads.py            # Thread count sweep
├── profile_decode_step.py      # Per-Linear decode profiler
└── profile_decode_modules.py   # Per-module decode profiler
```
