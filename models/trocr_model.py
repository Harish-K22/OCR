"""TrOCR (Microsoft) wrapper using HuggingFace transformers.

TrOCR is a line-level OCR model. For full-page images we split them into
horizontal strips (simple row segmentation) and run TrOCR on each strip.
We keep two checkpoints: one fine-tuned on printed text, one on handwritten.
The caller can choose via the `variant` constructor argument.
"""

import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .base import OCRModel
import config


class TrOCRModel(OCRModel):

    def __init__(self, variant: str = "printed"):
        self.variant = variant
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        model_name = (
            config.TROCR_PRINTED_MODEL
            if self.variant == "printed"
            else config.TROCR_HANDWRITTEN_MODEL
        )
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def extract_text(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        strips = self._split_into_lines(img)
        lines = []
        for strip in strips:
            pixel_values = self.processor(
                images=strip, return_tensors="pt"
            ).pixel_values
            generated_ids = self.model.generate(pixel_values, max_new_tokens=128)
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            if text.strip():
                lines.append(text.strip())
        return "\n".join(lines)

    def get_name(self) -> str:
        suffix = "printed" if self.variant == "printed" else "handwritten"
        return f"TrOCR ({suffix})"

    # ── simple horizontal strip segmentation ──────────────────
    @staticmethod
    def _split_into_lines(img: Image.Image, min_height: int = 30) -> list:
        """Split an image into horizontal strips based on white-space gaps."""
        gray = np.array(img.convert("L"))
        # Row projection: average pixel intensity per row
        row_mean = gray.mean(axis=1)
        threshold = 240  # near-white rows are gaps

        in_text = False
        start = 0
        strips = []
        for i, val in enumerate(row_mean):
            if val < threshold and not in_text:
                in_text = True
                start = i
            elif val >= threshold and in_text:
                in_text = False
                if i - start >= min_height:
                    strips.append(img.crop((0, start, img.width, i)))
        # Handle last strip
        if in_text and len(gray) - start >= min_height:
            strips.append(img.crop((0, start, img.width, len(gray))))

        # Fallback: if segmentation finds nothing, use the whole image
        if not strips:
            strips = [img]
        return strips
