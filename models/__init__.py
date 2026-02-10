from .base import OCRModel
from .tesseract_ocr import TesseractOCR
from .easy_ocr import EasyOCRModel
from .paddle_ocr import PaddleOCRModel
from .trocr_model import TrOCRModel
from .doctr_model import DocTRModel
from .mistral_ocr import MistralOCR

LOCAL_MODELS = [TesseractOCR, EasyOCRModel, PaddleOCRModel, TrOCRModel, DocTRModel]
API_MODELS = [MistralOCR]
ALL_MODELS = LOCAL_MODELS + API_MODELS
