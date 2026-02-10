"""EasyOCR wrapper."""

import easyocr

from .base import OCRModel


class EasyOCRModel(OCRModel):

    def __init__(self):
        self.reader = None

    def load_model(self) -> None:
        self.reader = easyocr.Reader(["en"], gpu=False)

    def extract_text(self, image_path: str) -> str:
        results = self.reader.readtext(image_path, detail=0, paragraph=True)
        return "\n".join(results).strip()

    def get_name(self) -> str:
        return "EasyOCR"
