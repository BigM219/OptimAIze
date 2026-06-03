import html
import logging

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from .base import BaseVLMBackend
from ..prompts import DOTS_MOCR_CATEGORY_PROMPTS as CATEGORY_PROMPTS
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


def _paddle_table_tokens_to_html(text: str) -> str:
    if "<fcel>" not in text and "<nl>" not in text:
        return text
    rows = []
    for raw_row in text.replace("<ecel>", "").split("<nl>"):
        cells = [c.strip() for c in raw_row.split("<fcel>") if c.strip()]
        if cells:
            rows.append(cells)
    if not rows:
        return text
    html_rows = []
    for row in rows:
        html_cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        html_rows.append(f"<tr>{html_cells}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


class PaddleOCRVLBackend(BaseVLMBackend):
    """Conservative CPU backend for PaddlePaddle/PaddleOCR-VL."""

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PaddleOCR-VL-1.6",
        device: str = "cpu",
        max_new_tokens: int = 1024,
        quantize_mode: str = "none",
        auto_runtime: RuntimeMode = "off",
        table_prompt: str = "fast",
    ):
        self.device = torch.device(device)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.table_prompt = table_prompt
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.runtime_policy: RuntimePolicy | None = None
        if auto_runtime != "off":
            profile = promoted_backend_profile("paddleocr-vl", model_id, config)
            self.runtime_policy = build_runtime_policy(profile, auto_runtime)
            if quantize_mode in ("auto", "selective"):
                quantize_mode = self.runtime_policy.quantize_mode
        self.quantize_mode = "mlp_lm_head" if quantize_mode == "selective" else quantize_mode
        if not hasattr(config, "text_config"):
            config.text_config = config

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except ValueError:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            except ValueError:
                self.model = AutoModel.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
        self._apply_cpu_optimizations()
        self.model.to(self.device)
        self.model.eval()
        logger.info("PaddleOCR-VL backend initialized on CPU")

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
            raise ValueError(f"Unsupported PaddleOCR-VL quantize_mode: {self.quantize_mode}")
        logger.info("PaddleOCR-VL CPU optimization quantize_mode=%s replaced=%d", self.quantize_mode, replaced)

    def supports_multi_image_single_call(self) -> bool:
        return True

    def _generate_multipage(self, images: list[Image.Image], instruction: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": build_multipage_content(images, instruction)}]
        max_pixels = 1280 * 28 * 28
        if hasattr(self.processor, "apply_chat_template"):
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "images_kwargs": {
                        "size": {
                            "shortest_edge": self.processor.image_processor.min_pixels,
                            "longest_edge": max_pixels,
                        }
                    }
                },
            )
        else:
            raise NotImplementedError("PaddleOCR-VL processor does not expose apply_chat_template")
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

    def _task_prompt(self, category: str) -> str:
        cat = category.strip().lower()
        use_fast_table_ocr = (
            cat == "table"
            and self.table_prompt == "fast"
            and self.model_id.lower().endswith("paddleocr-vl-1.6")
        )
        return "Table Recognition:" if cat == "table" and not use_fast_table_ocr else "OCR:"

    def _build_single_image_inputs(self, image: Image.Image, category: str):
        if image.mode != "RGB":
            image = image.convert("RGB")
        task_prompt = self._task_prompt(category)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": task_prompt}]}]
        max_pixels = 1280 * 28 * 28
        if hasattr(self.processor, "apply_chat_template"):
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "images_kwargs": {
                        "size": {
                            "shortest_edge": self.processor.image_processor.min_pixels,
                            "longest_edge": max_pixels,
                        }
                    }
                },
            )
        else:
            inputs = self.processor(images=image, text=task_prompt, return_tensors="pt")
        inputs.pop("mm_token_type_ids", None)
        return inputs

    def supports_visual_token_cache(self) -> bool:
        return True

    def build_visual_cache(self, image: Image.Image, category: str = "plain") -> dict[str, object]:
        inputs = self._build_single_image_inputs(image, category)
        tensors = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in inputs.items()}
        return {
            "backend": "paddleocr-vl",
            "model_id": self.model_id,
            "category": category,
            "table_prompt": self.table_prompt,
            "tensors": tensors,
        }

    @torch.inference_mode()
    def generate_ocr_from_visual_cache(self, cache: dict[str, object], category: str = "plain") -> str:
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in dict(cache["tensors"]).items()}
        return self._generate_ocr_from_inputs(inputs, category)

    @torch.inference_mode()
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        inputs = move_inputs_to_device(self._build_single_image_inputs(image, category), self.device)
        return self._generate_ocr_from_inputs(inputs, category)

    def _generate_ocr_from_inputs(self, inputs, category: str, generate_kwargs: dict[str, object] | None = None) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if generate_kwargs is None:
            cat = category.strip().lower()
            non_table_cap = self.runtime_policy.non_table_max_new_tokens if self.runtime_policy else 16
            table_cap = self.runtime_policy.table_max_new_tokens if self.runtime_policy else None
            if cat == "table":
                cap = min(self.max_new_tokens, table_cap) if table_cap is not None else self.max_new_tokens
            else:
                cap = self.max_new_tokens if non_table_cap is None else min(self.max_new_tokens, non_table_cap)
            generate_kwargs = generation_kwargs(tokenizer, cap)
        outputs = self.model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        end = -1 if outputs.shape[-1] > input_len else None
        generated = outputs[0][input_len:end]
        if tokenizer is not None:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            decoded = self.processor.decode(generated, skip_special_tokens=True)
        if category.strip().lower() == "table":
            decoded = _paddle_table_tokens_to_html(decoded)
        return parse_dots_mocr_output(decoded, category)
