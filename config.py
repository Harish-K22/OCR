"""
Configuration for OCR Models Comparison project.
Paths, API keys (from env vars), and model settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
SCORES_DIR = RESULTS_DIR / "scores"
VIS_DIR = RESULTS_DIR / "visualizations"

# ── Dataset Categories ────────────────────────────────────────
# Each category: (folder_name, display_label, num_samples)
CATEGORIES = {
    "printed":       ("printed",       "Printed Forms (FUNSD)",         6),
    "handwritten":   ("handwritten",   "Handwritten Lines (IAM)",       6),
    "receipts":      ("receipts",      "Receipts (SROIE)",              6),
    "scene_text":    ("scene_text",    "Scene / Street Text",           6),
    "dense_text":    ("dense_text",    "Dense Book Pages",              6),
    "degraded":      ("degraded",      "Degraded / Noisy Scans",        6),
}

def get_category_dirs(cat_key: str):
    """Return (images_dir, gt_dir) for a given category key."""
    folder = CATEGORIES[cat_key][0]
    images_dir = DATASETS_DIR / folder / "images"
    gt_dir = DATASETS_DIR / folder / "ground_truth"
    return images_dir, gt_dir

def get_category_label(cat_key: str) -> str:
    return CATEGORIES[cat_key][1]

# Create all category directories
for cat_key in CATEGORIES:
    img_dir, gt_dir = get_category_dirs(cat_key)
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ── Tesseract Path (Windows) ──────────────────────────────────
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ── Model Configs ─────────────────────────────────────────────
TROCR_PRINTED_MODEL = "microsoft/trocr-small-printed"
TROCR_HANDWRITTEN_MODEL = "microsoft/trocr-small-handwritten"

MISTRAL_MODEL = "pixtral-12b-2409"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# ── Dataset Config ────────────────────────────────────────────
SAMPLES_PER_CATEGORY = 6
