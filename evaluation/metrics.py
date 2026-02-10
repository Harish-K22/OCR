"""
Evaluation metrics for OCR model comparison.
CER, WER, accuracy, and timing utilities.
"""

import re
import time
from functools import wraps

from jiwer import cer, wer


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison: lowercase, collapse whitespace,
    strip leading/trailing whitespace, remove non-alphanumeric chars
    (except spaces)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_cer(prediction: str, reference: str) -> float:
    """Character Error Rate (lower is better). Returns 0-1 range."""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    return min(cer(ref, pred), 1.0)


def compute_wer(prediction: str, reference: str) -> float:
    """Word Error Rate (lower is better). Returns 0-1 range."""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    return min(wer(ref, pred), 1.0)


def compute_accuracy(prediction: str, reference: str) -> float:
    """Accuracy as 1 - CER, expressed as a percentage (0-100)."""
    return max(0.0, (1.0 - compute_cer(prediction, reference)) * 100)


def timed(func):
    """Decorator that returns (result, elapsed_seconds)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper
