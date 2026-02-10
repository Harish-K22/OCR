"""Tesseract OCR wrapper using pytesseract."""

import pytesseract
from PIL import Image

from .base import OCRModel
import config


class TesseractOCR(OCRModel):

    def load_model(self) -> None:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    def extract_text(self, image_path: str) -> str:
        img = Image.open(image_path)
        # --psm 3: Fully automatic page segmentation (default)
        # --psm 6: Assume a single uniform block of text
        text = pytesseract.image_to_string(img, config="--psm 3")
        return text.strip()

    def get_name(self) -> str:
        return "Tesseract"
