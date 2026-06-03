# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Adapted for modular CPU-only execution.

import logging
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image
from transformers import (
    AutoModelForObjectDetection,
    PPDocLayoutV3ImageProcessor,
)

logger = logging.getLogger(__name__)

# Target canvas size for PP-DocLayoutV3
_LAYOUT_TARGET_H, _LAYOUT_TARGET_W = 800, 800

# Layout categories to keep for OCR
LAYOUT_TO_OCR_CATEGORY = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "list-item": "list-item",
    "title": "title",
    "header": "text",
    "footer": "page-footer",
    "number": "text",
    "figure_title": "caption",
    "paragraph_title": "section-header",
    "doc_title": "title",
    "reference_content": "text",
    "reference": "text",
    "abstract": "text",
    "aside_text": "text",
    "content": "text",
    "formula_number": "text",
    "vision_footnote": "footnote",
    "algorithm": "text",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "section-header": "section-header",
    # Categories with no text to extract
    "image": None,
    "picture": None,
    "figure": None,
    "chart": None,
    "seal": None,
}

def _box_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def _intersection_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )

def _containment_ratio(small, large):
    area = _box_area(small)
    if area <= 0:
        return 0.0
    return _intersection_area(small, large) / area

def _iou(a, b):
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0

def dedup_overlapping_detections(
    detections: list[dict], iou_threshold: float = 0.8, area_ratio_threshold: float = 0.9,
) -> list[dict]:
    """Remove near-duplicate boxes (high IoU), keeping the larger one."""
    if len(detections) <= 1:
        return detections

    suppressed = set()
    for i in range(len(detections)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            if _iou(detections[i]["bbox"], detections[j]["bbox"]) > iou_threshold:
                area_i = _box_area(detections[i]["bbox"])
                area_j = _box_area(detections[j]["bbox"])
                ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 1.0
                if ratio > area_ratio_threshold:
                    loser = random.choice([i, j])
                    suppressed.add(loser)
                    if loser == i:
                        break
                elif area_i >= area_j:
                    suppressed.add(j)
                else:
                    suppressed.add(i)
                    break
    return [d for k, d in enumerate(detections) if k not in suppressed]

def filter_nested_detections(
    detections: list[dict], containment_threshold: float = 0.8
) -> list[dict]:
    """Remove any box that is mostly contained within a strictly larger box."""
    areas = [_box_area(d["bbox"]) for d in detections]
    keep = []
    for i, det in enumerate(detections):
        is_nested = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            if areas[j] <= areas[i]:
                continue
            if _containment_ratio(det["bbox"], other["bbox"]) > containment_threshold:
                is_nested = True
                break
        if not is_nested:
            keep.append(det)
    return keep


class PPDocLayoutDetector:
    """CPU-Optimized PP-DocLayoutV3 structure detector."""

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        logger.info(f"Loading PP-DocLayoutV3 on {device}...")
        
        # Load the image processor and model in float32 for CPU efficiency
        self.processor = PPDocLayoutV3ImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForObjectDetection.from_pretrained(
                model_id, torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def detect(self, image: Image.Image, threshold: float = 0.3) -> list[dict]:
        """Detect document layout regions on a single PIL image.

        Returns a list of dicts: {'category', 'bbox' [x1, y1, x2, y2], 'score'},
        sorted by reading order.
        """
        img_w, img_h = image.size
        target_size = torch.tensor([[img_h, img_w]])
        
        # CPU-optimized preprocessing: preserve aspect ratio on the detector canvas.
        tensor_img = tvF.pil_to_tensor(image).to(device=self.device, dtype=torch.float32)
        scale_factor = min(_LAYOUT_TARGET_W / img_w, _LAYOUT_TARGET_H / img_h)
        resized_w = max(1, round(img_w * scale_factor))
        resized_h = max(1, round(img_h * scale_factor))
        resized = F.interpolate(
            tensor_img.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )
        batch = torch.zeros(
            (1, 3, _LAYOUT_TARGET_H, _LAYOUT_TARGET_W),
            device=self.device,
            dtype=torch.float32,
        )
        pad_x = (_LAYOUT_TARGET_W - resized_w) // 2
        pad_y = (_LAYOUT_TARGET_H - resized_h) // 2
        batch[:, :, pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        pixel_batch = (batch.clamp_(0, 255) / 255.0)

        # Run forward pass
        outputs = self.model(pixel_values=pixel_batch)

        # Post-process outputs
        logits = outputs.logits
        boxes = outputs.pred_boxes
        order_logits = outputs.order_logits

        # Convert boxes from normalized center-width-height to xyxy coordinates
        box_centers, box_dims = boxes.split(2, dim=-1)
        boxes_xyxy = torch.cat(
            [box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1
        )

        # Remove canvas padding and scale boxes back to original image coordinates.
        canvas = torch.tensor(
            [
                pad_x / _LAYOUT_TARGET_W,
                pad_y / _LAYOUT_TARGET_H,
                pad_x / _LAYOUT_TARGET_W,
                pad_y / _LAYOUT_TARGET_H,
            ],
            device=self.device,
            dtype=boxes_xyxy.dtype,
        )
        content_scale = torch.tensor(
            [
                _LAYOUT_TARGET_W / resized_w,
                _LAYOUT_TARGET_H / resized_h,
                _LAYOUT_TARGET_W / resized_w,
                _LAYOUT_TARGET_H / resized_h,
            ],
            device=self.device,
            dtype=boxes_xyxy.dtype,
        )
        boxes_xyxy = (boxes_xyxy - canvas) * content_scale
        scale = torch.tensor([img_w, img_h, img_w, img_h], device=self.device, dtype=boxes_xyxy.dtype)
        boxes_xyxy = boxes_xyxy.clamp(0, 1) * scale

        num_queries = logits.shape[1]
        num_classes = logits.shape[2]
        scores = logits.sigmoid()
        
        scores_flat, index = scores.flatten(1).topk(num_queries, dim=-1)
        labels = index % num_classes
        box_indices = index // num_classes
        
        boxes_xyxy = boxes_xyxy.gather(
            dim=1, index=box_indices.unsqueeze(-1).expand(-1, -1, 4)
        )

        order_seqs = self.processor._get_order_seqs(order_logits)
        order_seqs = order_seqs.gather(dim=1, index=box_indices)

        # Process first (and only) image in the batch
        s_single = scores_flat[0]
        l_single = labels[0]
        b_single = boxes_xyxy[0]
        o_single = order_seqs[0]

        mask = s_single >= threshold
        o_valid = o_single[mask]
        _, indices_sorted = o_valid.sort()

        detections = []
        for si, li, bi in zip(
            s_single[mask][indices_sorted],
            l_single[mask][indices_sorted],
            b_single[mask][indices_sorted],
        ):
            detections.append({
                "category": self.id2label[li.item()],
                "bbox": [round(x, 2) for x in bi.tolist()],
                "score": round(si.item(), 4),
            })

        # Apply nested box filtering and duplicates deduplication
        filtered = filter_nested_detections(detections)
        dedupped = dedup_overlapping_detections(filtered)
        return dedupped
