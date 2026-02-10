"""DocTR (Mindee) OCR wrapper."""

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from .base import OCRModel


class DocTRModel(OCRModel):

    def __init__(self):
        self.predictor = None

    def load_model(self) -> None:
        self.predictor = ocr_predictor(pretrained=True)

    def extract_text(self, image_path: str) -> str:
        doc = DocumentFile.from_images(image_path)
        result = self.predictor(doc)
        # result.pages[0].blocks[].lines[].words[].value
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    lines.append(" ".join(words))
        return "\n".join(lines).strip()

    def get_name(self) -> str:
        return "DocTR"
