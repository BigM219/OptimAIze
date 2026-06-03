import logging

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from .base import BaseVLMBackend
from .dots_mocr.output_parser import parse_dots_mocr_output
from ..compute.int8_linear import replace_linears_with_int8
from ..runtime_policy import RuntimeMode, RuntimePolicy, build_runtime_policy, promoted_backend_profile
from .multi_page import (
    build_docqa_instruction,
    build_docqa_json_instruction,
    build_multipage_content,
    build_page_ocr_instruction,
    decode_generated,
    generation_kwargs,
    move_inputs_to_device,
    parse_page_json_array,
)

logger = logging.getLogger(__name__)


class GLMOCRBackend(BaseVLMBackend):
    """CPU backend for zai-org/GLM-OCR."""

    def __init__(
        self,
        model_id: str = "zai-org/GLM-OCR",
        device: str = "cpu",
        max_new_tokens: int = 1024,
        quantize_mode: str = "none",
        auto_runtime: RuntimeMode = "off",
    ):
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.runtime_policy: RuntimePolicy | None = None
        if auto_runtime != "off":
            profile = promoted_backend_profile("glm-ocr", model_id, config)
            self.runtime_policy = build_runtime_policy(profile, auto_runtime)
            if quantize_mode in ("auto", "selective"):
                quantize_mode = self.runtime_policy.quantize_mode
        self.quantize_mode = "mlp_lm_head" if quantize_mode == "selective" else quantize_mode
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._apply_cpu_optimizations()
        self.model.to(self.device)
        self.model.eval()
        logger.info("GLM-OCR backend initialized on CPU")

    def _apply_cpu_optimizations(self) -> None:
        if self.quantize_mode in ("none", "selective"):
            return
        if self.quantize_mode == "lm_head":
            replaced = replace_linears_with_int8(
                self.model,
                only_names=("lm_head",),
                keep_fp32_for_prefill=False,
            )
        elif self.quantize_mode == "mlp":
            replaced = replace_linears_with_int8(
                self.model,
                only_names=("language_model.layers", ".mlp."),
                skip_names=("self_attn", "visual", "vision"),
                keep_fp32_for_prefill=True,
            )
        elif self.quantize_mode == "mlp_lm_head":
            replaced = replace_linears_with_int8(
                self.model,
                only_names=("language_model.layers", ".mlp.", "lm_head"),
                skip_names=("self_attn", "visual", "vision"),
                keep_fp32_for_prefill=True,
            )
        else:
            raise ValueError(f"Unsupported GLM-OCR quantize_mode: {self.quantize_mode}")
        logger.info("GLM-OCR CPU optimization quantize_mode=%s replaced=%d", self.quantize_mode, replaced)

    def supports_multi_image_single_call(self) -> bool:
        return True

    def _generate_multipage(self, images: list[Image.Image], instruction: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": build_multipage_content(images, instruction)}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_inputs_to_device(inputs, self.device)
        tokenizer = getattr(self.processor, "tokenizer", None)
        outputs = self.model.generate(**inputs, **generation_kwargs(tokenizer, max_new_tokens))
        return decode_generated(self.processor, tokenizer, inputs, outputs)

    @torch.inference_mode()
    def generate_ocr_pages(self, images: list[Image.Image], category: str = "plain") -> list[str]:
        raw = self._generate_multipage(images, build_page_ocr_instruction(category), self.max_new_tokens)
        return parse_page_json_array(raw, len(images))

    @torch.inference_mode()
    def docqa_pages(
        self,
        images: list[Image.Image],
        question: str,
        max_new_tokens: int | None = None,
    ) -> str:
        return self._generate_multipage(
            images,
            build_docqa_json_instruction(question),
            max_new_tokens or self.max_new_tokens,
        )

    @torch.inference_mode()
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        cat = category.strip().lower()
        if cat == "formula":
            task_prompt = "Formula Recognition:"
        else:
            task_prompt = "Text Recognition:"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": task_prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        inputs.pop("token_type_ids", None)
        inputs.pop("mm_token_type_ids", None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        non_table_cap = self.runtime_policy.non_table_max_new_tokens if self.runtime_policy else 32
        cap = self.max_new_tokens if cat == "table" or non_table_cap is None else min(self.max_new_tokens, non_table_cap)
        generate_kwargs = {
            "max_new_tokens": cap,
            "do_sample": False,
            "use_cache": True,
        }
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        outputs = self.model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_len:]
        decoded = self.processor.decode(generated, skip_special_tokens=True)
        return parse_dots_mocr_output(decoded, category)
