import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Literal


RuntimeMode = Literal["off", "conservative", "speed", "experimental"]
QuantizeMode = Literal["none", "lm_head", "mlp", "mlp_lm_head"]


@dataclass(frozen=True)
class CPUProfile:
    logical_threads: int
    instruction_set: str
    fp32_vector_width: int
    int8_vector_width: int


@dataclass(frozen=True)
class ModelComputeProfile:
    backend: str
    model_id: str
    text_layers: int | None
    hidden_size: int | None
    intermediate_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    vocab_size: int | None
    vision_layers: int | None = None
    vision_hidden_size: int | None = None
    patch_size: int | None = None
    architecture_family: str = "unknown"
    output_style: str = "plain_ocr"
    correctness_status: str = "unknown"


@dataclass(frozen=True)
class RuntimeCostEstimate:
    qkv_cost: int
    mlp_cost: int
    lm_head_cost: int
    per_layer_cost: int
    decode_token_cost: int
    mlp_share: float
    lm_head_share: float
    kv_ratio: float


@dataclass(frozen=True)
class RuntimePolicy:
    mode: RuntimeMode
    backend: str
    model_id: str
    non_table_max_new_tokens: int | None
    table_max_new_tokens: int | None
    quantize_mode: QuantizeMode
    use_int8_decode_gemv: bool
    use_fused_mlp: bool
    use_int8_lm_head: bool
    suggested_threads: int | None
    interop_threads: int
    promoted: bool
    requires_gate: bool
    reason: str
    cpu: CPUProfile
    profile: ModelComputeProfile
    cost: RuntimeCostEstimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_cpu_profile() -> CPUProfile:
    logical = os.cpu_count() or 1
    machine = " ".join(filter(None, (platform.processor(), platform.machine()))).lower()
    instruction_set = "scalar"
    fp32_width = 4
    int8_width = 16
    if "avx512" in machine:
        instruction_set = "avx512"
        fp32_width = 16
        int8_width = 64
    elif "amd64" in machine or "x86_64" in machine or "intel" in machine or "amd" in machine:
        instruction_set = "avx2_candidate"
        fp32_width = 8
        int8_width = 32
    return CPUProfile(
        logical_threads=logical,
        instruction_set=instruction_set,
        fp32_vector_width=fp32_width,
        int8_vector_width=int8_width,
    )


def _get_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if obj is None:
            return None
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
        if isinstance(obj, dict) and obj.get(name) is not None:
            return obj[name]
    return None


def _nested_config(config: Any, *names: str) -> Any:
    for name in names:
        value = _get_attr(config, name)
        if value is not None:
            return value
    return None


def extract_model_compute_profile(
    config: Any,
    backend: str,
    model_id: str,
    output_style: str = "plain_ocr",
    correctness_status: str = "unknown",
) -> ModelComputeProfile:
    text_config = _nested_config(config, "text_config", "llm_config", "language_config") or config
    vision_config = _nested_config(config, "vision_config", "visual_config", "image_config")
    text_layers = _get_attr(text_config, "num_hidden_layers", "n_layers", "num_layers")
    hidden_size = _get_attr(text_config, "hidden_size", "n_embd", "dim", "d_model")
    intermediate_size = _get_attr(text_config, "intermediate_size", "ffn_dim", "mlp_dim")
    num_heads = _get_attr(text_config, "num_attention_heads", "n_heads")
    num_kv_heads = _get_attr(text_config, "num_key_value_heads", "n_kv_heads") or num_heads
    vocab_size = _get_attr(text_config, "vocab_size") or _get_attr(config, "vocab_size")
    vision_layers = _get_attr(vision_config, "num_hidden_layers", "n_layers", "num_layers")
    vision_hidden = _get_attr(vision_config, "hidden_size", "dim", "d_model")
    patch_size = _get_attr(vision_config, "patch_size", "spatial_patch_size")
    architecture_family = str(_get_attr(text_config, "model_type") or _get_attr(config, "model_type") or "unknown")
    return ModelComputeProfile(
        backend=backend,
        model_id=model_id,
        text_layers=int(text_layers) if text_layers is not None else None,
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        intermediate_size=int(intermediate_size) if intermediate_size is not None else None,
        num_attention_heads=int(num_heads) if num_heads is not None else None,
        num_key_value_heads=int(num_kv_heads) if num_kv_heads is not None else None,
        vocab_size=int(vocab_size) if vocab_size is not None else None,
        vision_layers=int(vision_layers) if vision_layers is not None else None,
        vision_hidden_size=int(vision_hidden) if vision_hidden is not None else None,
        patch_size=int(patch_size) if patch_size is not None else None,
        architecture_family=architecture_family,
        output_style=output_style,
        correctness_status=correctness_status,
    )


def estimate_runtime_cost(profile: ModelComputeProfile) -> RuntimeCostEstimate:
    h = profile.hidden_size or 0
    i = profile.intermediate_size or 0
    l = profile.text_layers or 0
    heads = profile.num_attention_heads or 1
    kv_heads = profile.num_key_value_heads or heads
    vocab = profile.vocab_size or 0
    kv_ratio = kv_heads / heads if heads else 1.0
    qkv_cost = int(h * (h + 2 * h * kv_ratio)) if h else 0
    mlp_cost = int(3 * h * i) if h and i else 0
    lm_head_cost = int(h * vocab) if h and vocab else 0
    per_layer_cost = qkv_cost + mlp_cost + h * h
    decode_token_cost = l * per_layer_cost + lm_head_cost
    if decode_token_cost > 0:
        mlp_share = (l * mlp_cost) / decode_token_cost
        lm_head_share = lm_head_cost / decode_token_cost
    else:
        mlp_share = 0.0
        lm_head_share = 0.0
    return RuntimeCostEstimate(
        qkv_cost=qkv_cost,
        mlp_cost=mlp_cost,
        lm_head_cost=lm_head_cost,
        per_layer_cost=per_layer_cost,
        decode_token_cost=decode_token_cost,
        mlp_share=mlp_share,
        lm_head_share=lm_head_share,
        kv_ratio=kv_ratio,
    )


def _suggest_threads(cost: RuntimeCostEstimate, cpu: CPUProfile) -> int:
    work = max(cost.mlp_cost, cost.lm_head_cost, cost.per_layer_cost)
    if work < 1_000_000:
        return max(1, min(4, cpu.logical_threads))
    if work < 8_000_000:
        return max(1, min(8, cpu.logical_threads))
    return max(1, cpu.logical_threads)


def _choose_cap(profile: ModelComputeProfile, cost: RuntimeCostEstimate) -> int | None:
    backend = profile.backend
    if backend == "falcon-ocr":
        return None
    if profile.output_style == "html_like" or backend == "surya-ocr":
        return 64
    if backend == "lighton-ocr":
        return 8
    if profile.output_style == "short_text" or backend == "paddleocr-vl":
        return 16
    if backend == "glm-ocr":
        return 32
    if cost.decode_token_cost > 500_000_000:
        return 16
    if (profile.hidden_size or 0) >= 1536:
        return 32
    return 32


def _choose_quantize_mode(profile: ModelComputeProfile, cost: RuntimeCostEstimate, mode: RuntimeMode) -> QuantizeMode:
    if mode == "conservative" and profile.backend in {"glm-ocr", "paddleocr-vl"}:
        return "mlp_lm_head"
    if mode in ("off", "conservative"):
        return "none"
    if profile.correctness_status == "failed":
        return "none"
    if profile.backend in {"lighton-ocr", "surya-ocr"} and mode != "experimental":
        return "none"
    mlp = cost.mlp_share >= 0.45 or cost.mlp_cost >= 8_000_000
    lm_head = cost.lm_head_share >= 0.12 or cost.lm_head_cost >= 80_000_000
    if mlp and lm_head:
        return "mlp_lm_head"
    if mlp:
        return "mlp"
    if lm_head:
        return "lm_head"
    return "none"


def build_runtime_policy(
    profile: ModelComputeProfile,
    mode: RuntimeMode = "conservative",
    cpu: CPUProfile | None = None,
) -> RuntimePolicy:
    cpu = cpu or inspect_cpu_profile()
    cost = estimate_runtime_cost(profile)
    non_table_cap = _choose_cap(profile, cost)
    quantize_mode = _choose_quantize_mode(profile, cost, mode)
    promoted = mode == "conservative" and profile.correctness_status != "failed"
    if profile.backend == "dots-mocr" and profile.correctness_status == "failed":
        promoted = False
    use_int8_lm_head = quantize_mode in {"lm_head", "mlp_lm_head"}
    use_fused_mlp = mode in {"speed", "experimental"} and quantize_mode in {"mlp", "mlp_lm_head"}
    reason = "architecture_policy"
    if mode == "conservative":
        reason = "promoted_speed_defaults" if quantize_mode != "none" else "promoted_safe_defaults"
    if profile.correctness_status == "failed":
        reason = "correctness_gate_failed"
    threads = None if mode == "off" else _suggest_threads(cost, cpu)
    if profile.backend == "lighton-ocr":
        table_cap = 34
    elif profile.backend == "paddleocr-vl" and profile.model_id.lower().endswith("paddleocr-vl-1.6"):
        table_cap = 32
    else:
        table_cap = None
    return RuntimePolicy(
        mode=mode,
        backend=profile.backend,
        model_id=profile.model_id,
        non_table_max_new_tokens=non_table_cap,
        table_max_new_tokens=table_cap,
        quantize_mode=quantize_mode,
        use_int8_decode_gemv=quantize_mode != "none",
        use_fused_mlp=use_fused_mlp,
        use_int8_lm_head=use_int8_lm_head,
        suggested_threads=threads,
        interop_threads=max(1, min(threads or cpu.logical_threads, 4)),
        promoted=promoted,
        requires_gate=mode in {"speed", "experimental"},
        reason=reason,
        cpu=cpu,
        profile=profile,
        cost=cost,
    )


def promoted_backend_profile(backend: str, model_id: str, config: Any) -> ModelComputeProfile:
    styles = {
        "falcon-ocr": "plain_ocr",
        "paddleocr-vl": "short_text",
        "glm-ocr": "plain_ocr",
        "lighton-ocr": "short_text",
        "surya-ocr": "html_like",
        "dots-mocr": "table_heavy",
    }
    statuses = {"dots-mocr": "failed"}
    return extract_model_compute_profile(
        config,
        backend=backend,
        model_id=model_id,
        output_style=styles.get(backend, "plain_ocr"),
        correctness_status=statuses.get(backend, "passed"),
    )
