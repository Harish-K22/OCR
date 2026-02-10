"""Abstract base class for all OCR models."""

from abc import ABC, abstractmethod


class OCRModel(ABC):
    """Base interface that every OCR model wrapper must implement."""

    @abstractmethod
    def load_model(self) -> None:
        """Load / initialize the model (weights, reader objects, etc.)."""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Run OCR on a single image and return the extracted text."""

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable model name for charts and tables."""
