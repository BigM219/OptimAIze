# Backward-compatibility shim. The real implementation lives in
# `optimaize_ocr.compute.quantization` and is shared between backends.

from ...compute.quantization import (
    quantize_model_layer_by_layer,
    quantize_decoder_layers_inplace,
    quantize_submodule_inplace,
)

__all__ = [
    "quantize_model_layer_by_layer",
    "quantize_decoder_layers_inplace",
    "quantize_submodule_inplace",
]
