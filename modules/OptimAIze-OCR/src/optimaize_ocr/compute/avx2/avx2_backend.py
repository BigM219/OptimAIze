import os
import sys
import ctypes
import subprocess
import logging

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_PATH = os.path.join(BASE_DIR, "avx2_gemv.cpp")
DLL_NAME = "avx2_gemv.dll" if sys.platform == "win32" else "libavx2_gemv.so"
DLL_PATH = os.path.join(BASE_DIR, DLL_NAME)

AVX2_CPP_AVAILABLE = False
_lib = None


def _dll_is_stale() -> bool:
    """True if the compiled DLL is missing or older than the source .cpp."""
    if not os.path.exists(DLL_PATH):
        return True
    try:
        return os.path.getmtime(CPP_PATH) > os.path.getmtime(DLL_PATH)
    except OSError:
        return True


def _resolve_compiler(name: str) -> str | None:
    """Find a C++ compiler binary, searching common conda + system paths."""
    import shutil
    # First check PATH
    p = shutil.which(name)
    if p:
        return p
    # Common conda mingw locations (Windows)
    if sys.platform == "win32" and name in ("g++", "gcc"):
        candidates = []
        env_prefix = os.environ.get("CONDA_PREFIX")
        if env_prefix:
            candidates.append(os.path.join(env_prefix, "Library", "mingw-w64", "bin", f"{name}.exe"))
        # Walk through known conda installs
        for root in (os.path.expanduser("~/miniconda3"),
                     os.path.expanduser("~/anaconda3"),
                     "C:/miniconda3", "C:/anaconda3"):
            if not os.path.isdir(root):
                continue
            for env in ("Library/mingw-w64/bin", "envs/base_py311/Library/mingw-w64/bin"):
                cand = os.path.join(root, env, f"{name}.exe")
                candidates.append(cand)
        for c in candidates:
            if os.path.isfile(c):
                return c
    return None


def _add_windows_dll_dirs() -> list[object]:
    if sys.platform != "win32":
        return []
    dirs = {BASE_DIR}
    for compiler in ("g++", "gcc", "clang++"):
        resolved = _resolve_compiler(compiler)
        if resolved:
            dirs.add(os.path.dirname(resolved))
    handles = []
    for directory in dirs:
        if not directory or not os.path.isdir(directory):
            continue
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                handles.append(os.add_dll_directory(directory))
            except OSError:
                pass
    return handles


def _load_avx2_library():
    _add_windows_dll_dirs()
    return ctypes.CDLL(DLL_PATH)


def compile_cpp_library():
    """Compile (or reuse) the AVX2 C++ intrinsics library.

    The compiled DLL is reused only when its mtime is newer than the source .cpp;
    otherwise it is recompiled so kernel updates are picked up automatically.
    """
    global AVX2_CPP_AVAILABLE, _lib

    if not _dll_is_stale():
        try:
            _lib = _load_avx2_library()
            AVX2_CPP_AVAILABLE = True
            logger.info(f"Loaded existing precompiled AVX2 C++ Intrinsics library from: {DLL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load existing AVX2 C++ library: {e}. Recompiling...")
            try:
                os.remove(DLL_PATH)
            except Exception:
                pass
    else:
        if os.path.exists(DLL_PATH):
            logger.info("AVX2 source is newer than compiled DLL — rebuilding.")
            try:
                os.remove(DLL_PATH)
            except Exception:
                pass

    if not os.path.exists(CPP_PATH):
        logger.error(f"C++ source file not found at: {CPP_PATH}")
        return False

    logger.info("Attempting to compile C++ Intrinsics with AVX2 + OpenMP support...")

    compile_success = False

    # 1. Try GCC (g++) — search PATH and known conda locations
    gpp = _resolve_compiler("g++")
    if gpp:
        try:
            cmd = [gpp, "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared",
                   "-funroll-loops", "-ffast-math", CPP_PATH, "-o", DLL_PATH]
            logger.info(f"Trying command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            compile_success = True
            logger.info("C++ AVX2 library compiled successfully using g++!")
        except subprocess.CalledProcessError as e:
            logger.warning(f"g++ failed: {e.stderr.decode(errors='ignore')[:500]}")
        except Exception as e:
            logger.debug(f"g++ exec error: {e}")

    if not compile_success:
        clangpp = _resolve_compiler("clang++")
        if clangpp:
            try:
                cmd = [clangpp, "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared",
                       "-funroll-loops", "-ffast-math", CPP_PATH, "-o", DLL_PATH]
                logger.info(f"Trying command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, capture_output=True)
                compile_success = True
                logger.info("C++ AVX2 library compiled successfully using clang++!")
            except subprocess.CalledProcessError as e:
                logger.warning(f"clang++ failed: {e.stderr.decode(errors='ignore')[:500]}")
            except Exception as e:
                logger.debug(f"clang++ exec error: {e}")

    if not compile_success:
        cl = _resolve_compiler("cl") or _resolve_compiler("cl.exe")
        if cl:
            try:
                cmd = [cl, "/O2", "/arch:AVX2", "/openmp", "/fp:fast", "/LD",
                       CPP_PATH, f"/Fe:{DLL_PATH}"]
                logger.info(f"Trying command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, capture_output=True)
                compile_success = True
                logger.info("C++ AVX2 library compiled successfully using MSVC cl.exe!")
                # Cleanup MSVC temporary compiler artifacts
                for ext in [".obj", ".lib", ".exp"]:
                    temp_file = os.path.join(BASE_DIR, f"avx2_gemv{ext}")
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass
            except subprocess.CalledProcessError as e:
                logger.warning(f"cl.exe failed: {e.stderr.decode(errors='ignore')[:500]}")
            except Exception as e:
                logger.debug(f"cl.exe exec error: {e}")

    if compile_success:
        try:
            _lib = _load_avx2_library()
            AVX2_CPP_AVAILABLE = True
            logger.info("Successfully loaded C++ Intrinsics AVX2 DLL!")
            return True
        except Exception as e:
            logger.error(f"Compiled successfully but failed to load DLL: {e}")
            AVX2_CPP_AVAILABLE = False
            return False
    else:
        logger.warning(
            "C++ Compiler (g++, clang++, or cl.exe) not found in PATH or compilation failed. "
            "OCR pipeline will automatically fall back to Numba JIT high-performance AVX2 kernels "
            "using LLVM auto-vectorization and parallel thread scheduling. Zero performance loss!"
        )
        AVX2_CPP_AVAILABLE = False
        return False


# Trigger compilation/loading immediately
compile_cpp_library()

# Setup ctypes argument and return types if loaded
if AVX2_CPP_AVAILABLE and _lib is not None:
    _INT8_GEMV_ARGS = [
        ctypes.POINTER(ctypes.c_int8),   # w_int8
        ctypes.c_float,                  # scale
        ctypes.c_int,                    # zero_point
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,                    # out_features
        ctypes.c_int                     # in_features
    ]
    _INT8_GEMV_PC_ARGS = [
        ctypes.POINTER(ctypes.c_int8),   # w_int8
        ctypes.POINTER(ctypes.c_float),  # scales [out_features]
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,                    # out_features
        ctypes.c_int                     # in_features
    ]
    _FP32_GEMV_ARGS = [
        ctypes.POINTER(ctypes.c_float),  # w_float32
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,                    # out_features
        ctypes.c_int                     # in_features
    ]

    _FUSED_RMSNORM_GEMV_ARGS = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # x_norm_buf
        ctypes.POINTER(ctypes.c_float),  # w_float32 or w_int8
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,                    # out_features
        ctypes.c_int,                    # dim
        ctypes.c_float,                  # eps
    ]

    _FUSED_RMSNORM_GEMV_PC_ARGS = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # x_norm_buf
        ctypes.POINTER(ctypes.c_int8),   # w_int8
        ctypes.POINTER(ctypes.c_float),  # scales
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,                    # out_features
        ctypes.c_int,                    # dim
        ctypes.c_float,                  # eps
    ]

    # Fused SwiGLU MLP gate+up projection (per-channel symmetric INT8)
    _SWIGLU_PC_ARGS = [
        ctypes.POINTER(ctypes.c_int8),   # w_gate
        ctypes.POINTER(ctypes.c_float),  # scales_gate
        ctypes.POINTER(ctypes.c_float),  # bias_gate (may be NULL)
        ctypes.POINTER(ctypes.c_int8),   # w_up
        ctypes.POINTER(ctypes.c_float),  # scales_up
        ctypes.POINTER(ctypes.c_float),  # bias_up (may be NULL)
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,                    # hidden
        ctypes.c_int,                    # in_features
    ]

    # Fused QKV projection (per-channel symmetric INT8) for GQA attention
    _QKV_PC_ARGS = [
        ctypes.POINTER(ctypes.c_int8),   # w_q
        ctypes.POINTER(ctypes.c_float),  # scales_q
        ctypes.POINTER(ctypes.c_float),  # bias_q (may be NULL)
        ctypes.POINTER(ctypes.c_int8),   # w_k
        ctypes.POINTER(ctypes.c_float),  # scales_k
        ctypes.POINTER(ctypes.c_float),  # bias_k (may be NULL)
        ctypes.POINTER(ctypes.c_int8),   # w_v
        ctypes.POINTER(ctypes.c_float),  # scales_v
        ctypes.POINTER(ctypes.c_float),  # bias_v (may be NULL)
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # out_q
        ctypes.POINTER(ctypes.c_float),  # out_k
        ctypes.POINTER(ctypes.c_float),  # out_v
        ctypes.c_int,                    # q_out
        ctypes.c_int,                    # kv_out
        ctypes.c_int,                    # in_features
    ]

    # Fused RMSNorm + QKV projection
    _RMSNORM_QKV_ARGS = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # gamma (RMSNorm weight)
        ctypes.c_float,                  # eps
        ctypes.POINTER(ctypes.c_float),  # norm_buf (scratch)
        ctypes.POINTER(ctypes.c_int8),   # w_q
        ctypes.POINTER(ctypes.c_float),  # scales_q
        ctypes.POINTER(ctypes.c_float),  # bias_q
        ctypes.POINTER(ctypes.c_int8),   # w_k
        ctypes.POINTER(ctypes.c_float),  # scales_k
        ctypes.POINTER(ctypes.c_float),  # bias_k
        ctypes.POINTER(ctypes.c_int8),   # w_v
        ctypes.POINTER(ctypes.c_float),  # scales_v
        ctypes.POINTER(ctypes.c_float),  # bias_v
        ctypes.POINTER(ctypes.c_float),  # out_q
        ctypes.POINTER(ctypes.c_float),  # out_k
        ctypes.POINTER(ctypes.c_float),  # out_v
        ctypes.c_int,                    # q_out
        ctypes.c_int,                    # kv_out
        ctypes.c_int,                    # in_features
    ]

    _BINDINGS = [
        ("gemv_int8_avx2",              _INT8_GEMV_ARGS),
        ("gemv_int8_avx2_m4",           _INT8_GEMV_ARGS),
        ("gemv_int8_avx2_per_channel",     _INT8_GEMV_PC_ARGS),
        ("gemv_int8_avx2_per_channel_m4",  _INT8_GEMV_PC_ARGS),
        ("gemv_float32_avx2",              _FP32_GEMV_ARGS),
        ("gemv_float32_avx2_m4",        _FP32_GEMV_ARGS),
        ("rmsnorm_gemm_float32_avx2",          _FUSED_RMSNORM_GEMV_ARGS),
        ("rmsnorm_gemm_int8_per_channel_avx2", _FUSED_RMSNORM_GEMV_PC_ARGS),
        ("swiglu_gate_up_int8_per_channel_avx2", _SWIGLU_PC_ARGS),
        ("qkv_int8_per_channel_avx2",   _QKV_PC_ARGS),
        ("rmsnorm_qkv_int8_per_channel_avx2", _RMSNORM_QKV_ARGS),
    ]

    for fname, args in _BINDINGS:
        try:
            fn = getattr(_lib, fname)
            fn.argtypes = args
            fn.restype = None
        except Exception as e:
            # The multi-row kernels are optional; only the base ones must exist
            if fname in ("gemv_int8_avx2", "gemv_float32_avx2"):
                logger.error(f"Error binding {fname}: {e}")
                AVX2_CPP_AVAILABLE = False
            else:
                logger.warning(f"Optional kernel {fname} not found in DLL: {e}")


def cpp_gemv_int8(w_int8, scale, zero_point, x, bias, out):
    """Wrapper calling the compiled C++ AVX2 INT8 quantized GEMV.

    Uses the 4-row blocked kernel when out_features is divisible by 4 for
    better x-load reuse; otherwise falls back to the single-row kernel.
    """
    out_features, in_features = w_int8.shape
    w_ptr = w_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if hasattr(_lib, "gemv_int8_avx2_m4") and (out_features & 3) == 0 and out_features >= 32:
        _lib.gemv_int8_avx2_m4(
            w_ptr, scale, zero_point, x_ptr, bias_ptr, out_ptr,
            out_features, in_features
        )
    else:
        _lib.gemv_int8_avx2(
            w_ptr, scale, zero_point, x_ptr, bias_ptr, out_ptr,
            out_features, in_features
        )


def cpp_gemv_float32(w_float32, x, bias, out):
    """Wrapper calling the compiled C++ AVX2 float32 GEMV.

    Uses the 4-row blocked kernel when out_features is divisible by 4 for
    better x-load reuse; otherwise falls back to the single-row kernel.
    """
    out_features, in_features = w_float32.shape
    w_ptr = w_float32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if hasattr(_lib, "gemv_float32_avx2_m4") and (out_features & 3) == 0 and out_features >= 32:
        _lib.gemv_float32_avx2_m4(
            w_ptr, x_ptr, bias_ptr, out_ptr, out_features, in_features
        )
    else:
        _lib.gemv_float32_avx2(
            w_ptr, x_ptr, bias_ptr, out_ptr, out_features, in_features
        )


def cpp_gemv_int8_per_channel(w_int8, scales, x, bias, out):
    """C++ AVX2 per-channel symmetric INT8 GEMV wrapper.

    Per-row scales (zero-point implicit 0). All inputs must be C-contiguous
    float32 (except w_int8 which is int8).
    """
    out_features, in_features = w_int8.shape
    _lib.gemv_int8_avx2_per_channel(
        w_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_features, in_features,
    )


def has_cpp_per_channel() -> bool:
    """True iff the loaded AVX2 DLL exports the per-channel INT8 kernel."""
    return AVX2_CPP_AVAILABLE and _lib is not None and hasattr(_lib, "gemv_int8_avx2_per_channel")


def has_fused_rmsnorm_gemm() -> bool:
    """True iff the loaded AVX2 DLL exports the fused RMSNorm+GEMV kernels."""
    return AVX2_CPP_AVAILABLE and _lib is not None and hasattr(_lib, "rmsnorm_gemm_int8_per_channel_avx2")


def cpp_rmsnorm_gemm_int8_per_channel(x, w_int8, scales, bias, out, x_norm_buf, eps=1e-5):
    """Fused RMSNorm + per-channel INT8 GEMV.

    Avoids intermediate tensor allocation and extra memory read/write.
    """
    out_features, in_features = w_int8.shape
    _lib.rmsnorm_gemm_int8_per_channel_avx2(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_norm_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_features, in_features, eps
    )


def cpp_rmsnorm_gemm_float32(x, w_float32, bias, out, x_norm_buf, eps=1e-5):
    """Fused RMSNorm + float32 GEMV.

    Avoids intermediate tensor allocation and extra memory read/write.
    """
    out_features, in_features = w_float32.shape
    _lib.rmsnorm_gemm_float32_avx2(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_norm_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_float32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_features, in_features, eps
    )


def has_swiglu_int8_per_channel() -> bool:
    """True iff the loaded AVX2 DLL exports the fused SwiGLU MLP kernel."""
    return (AVX2_CPP_AVAILABLE
            and _lib is not None
            and hasattr(_lib, "swiglu_gate_up_int8_per_channel_avx2"))


_NULL_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)()


def cpp_swiglu_gate_up_int8_per_channel(
    w_gate, scales_gate, bias_gate,  # bias_* may be None
    w_up,   scales_up,   bias_up,
    x, out,
):
    """Fused SwiGLU gate+up projection (per-channel symmetric INT8).

    Computes ``out = silu(w_gate @ x + bias_gate) * (w_up @ x + bias_up)``
    in a single AVX2/OpenMP pass that reads ``x`` once for both matrices.

    Shapes: ``w_gate`` and ``w_up`` are int8 ``[hidden, in_features]``,
    ``scales_*`` are float32 ``[hidden]``, ``x`` is float32 ``[in_features]``,
    ``out`` is float32 ``[hidden]``. Biases are optional — pass ``None`` to
    skip them.
    """
    hidden, in_features = w_gate.shape
    bias_g_ptr = (bias_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                  if bias_gate is not None else _NULL_FLOAT_PTR)
    bias_u_ptr = (bias_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                  if bias_up is not None else _NULL_FLOAT_PTR)
    _lib.swiglu_gate_up_int8_per_channel_avx2(
        w_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias_g_ptr,
        w_up.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias_u_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        hidden, in_features,
    )


def has_qkv_int8_per_channel() -> bool:
    """True iff the loaded AVX2 DLL exports the fused QKV projection kernel."""
    return (AVX2_CPP_AVAILABLE
            and _lib is not None
            and hasattr(_lib, "qkv_int8_per_channel_avx2"))


def cpp_qkv_int8_per_channel(
    w_q, scales_q, bias_q,
    w_k, scales_k, bias_k,
    w_v, scales_v, bias_v,
    x, out_q, out_k, out_v,
):
    """Fused per-channel INT8 GEMV for q/k/v projections under GQA.

    All three matrices share the same input ``x`` (length ``in_features``).
    Biases may be ``None``. Output buffers must already exist with the
    correct length — ``out_q`` is ``[q_out]``, ``out_k`` and ``out_v`` are
    ``[kv_out]``.
    """
    q_out, in_features = w_q.shape
    kv_out, _ = w_k.shape
    bq = bias_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_q is not None else _NULL_FLOAT_PTR
    bk = bias_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_k is not None else _NULL_FLOAT_PTR
    bv = bias_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_v is not None else _NULL_FLOAT_PTR
    _lib.qkv_int8_per_channel_avx2(
        w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bq,
        w_k.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bk,
        w_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bv,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_out, kv_out, in_features,
    )


def has_rmsnorm_qkv_int8_per_channel() -> bool:
    """True iff the loaded AVX2 DLL exports the fused RMSNorm+QKV kernel."""
    return (AVX2_CPP_AVAILABLE
            and _lib is not None
            and hasattr(_lib, "rmsnorm_qkv_int8_per_channel_avx2"))


def cpp_rmsnorm_qkv_int8_per_channel(
    x, gamma, eps, norm_buf,
    w_q, scales_q, bias_q,
    w_k, scales_k, bias_k,
    w_v, scales_v, bias_v,
    out_q, out_k, out_v,
):
    """Fused RMSNorm + per-channel INT8 QKV projection under GQA.

    Folds ``RMSNorm(x, gamma)`` into the QKV fan-out so the normalized
    activations stay in-cache between the norm pass and the matmul.
    Caller must supply ``norm_buf`` as a contiguous float scratch buffer
    of length ``in_features`` — it will be overwritten with the
    normalized activations on every call.
    """
    q_out, in_features = w_q.shape
    kv_out, _ = w_k.shape
    bq = bias_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_q is not None else _NULL_FLOAT_PTR
    bk = bias_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_k is not None else _NULL_FLOAT_PTR
    bv = bias_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_v is not None else _NULL_FLOAT_PTR
    _lib.rmsnorm_qkv_int8_per_channel_avx2(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(eps),
        norm_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bq,
        w_k.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bk,
        w_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bv,
        out_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_out, kv_out, in_features,
    )
