"""PaddleOCR wrapper (v2.8.x API)."""

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR as _PaddleOCR

from .base import OCRModel


class PaddleOCRModel(OCRModel):

    def __init__(self):
        self.ocr = None

    def load_model(self) -> None:
        self.ocr = _PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def extract_text(self, image_path: str) -> str:
        result = self.ocr.ocr(image_path, cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]  # (bbox, (text, confidence))
                lines.append(text)
        return "\n".join(lines).strip()

    def get_name(self) -> str:
        return "PaddleOCR"
