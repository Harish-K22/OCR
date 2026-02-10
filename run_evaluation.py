"""
Main evaluation orchestrator.
Runs all LOCAL models (Tesseract, EasyOCR, PaddleOCR, TrOCR, DocTR) on ALL
dataset categories, computes metrics, saves CSVs, and generates visualizations.

Each model runs in a separate subprocess to avoid memory accumulation.

Usage:
    python run_evaluation.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import config
from evaluation.visualize import generate_all_visualizations


PYTHON = str(Path(config.BASE_DIR) / "venv" / "Scripts" / "python.exe")
WORKER = str(Path(config.BASE_DIR) / "_run_single_model.py")

# Model specs: (module_path, class_name, constructor_kwargs)
MODEL_SPECS = [
    ("models.tesseract_ocr", "TesseractOCR", {}),
    ("models.easy_ocr", "EasyOCRModel", {}),
    ("models.paddle_ocr", "PaddleOCRModel", {}),
    ("models.trocr_model", "TrOCRModel", {"variant": "printed"}),
    ("models.trocr_model", "TrOCRModel", {"variant": "handwritten"}),
    ("models.doctr_model", "DocTRModel", {}),
]


def main() -> None:
    csv_path = config.SCORES_DIR / "all_results.csv"
    all_results = []

    # Show dataset summary first
    print("=" * 60)
    print("DATASET CATEGORIES")
    print("=" * 60)
    for cat_key in config.CATEGORIES:
        img_dir, gt_dir = config.get_category_dirs(cat_key)
        n_img = len(list(img_dir.glob("*.png")))
        label = config.get_category_label(cat_key)
        print(f"  {label}: {n_img} images")
    print()

    for module_path, class_name, kwargs in MODEL_SPECS:
        spec_json = json.dumps({
            "module": module_path,
            "cls": class_name,
            "kwargs": kwargs,
        })
        print(f"\n{'='*60}")
        print(f"Running {class_name} {kwargs} in subprocess ...")
        print(f"{'='*60}")

        try:
            result = subprocess.run(
                [PYTHON, WORKER, spec_json],
                capture_output=True,
                text=True,
                timeout=900,
                cwd=str(config.BASE_DIR),
                env={**os.environ, "PYTHONIOENCODING": "utf-8",
                     "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                     "FLAGS_enable_pir_api": "0",
                     "FLAGS_use_mkldnn": "0"},
            )
            if result.stderr:
                for line in result.stderr.splitlines():
                    if "Error" in line or "FAIL" in line or "Traceback" in line:
                        print(f"  {line}")

            if result.returncode != 0:
                print(f"  [FAIL] Subprocess exited with code {result.returncode}")
                for line in result.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        data = json.loads(line[7:])
                        all_results.extend(data)
                continue

            for line in result.stdout.splitlines():
                if line.startswith("RESULT:"):
                    data = json.loads(line[7:])
                    all_results.extend(data)
                    print(f"  [OK] Got {len(data)} results")
                elif line.startswith("INFO:"):
                    print(f"  {line[5:]}")
        except subprocess.TimeoutExpired:
            print(f"  [FAIL] Subprocess timed out after 900s")
        except Exception as e:
            print(f"  [FAIL] {e}")

    # Save results
    if not all_results:
        print("\nNo results collected. Check errors above.")
        sys.exit(1)

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path} ({len(df)} rows)")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = df.groupby(["model", "category"]).agg(
        avg_cer=("cer", "mean"),
        avg_wer=("wer", "mean"),
        avg_acc=("accuracy", "mean"),
        avg_time=("time_sec", "mean"),
    ).round(3)
    print(summary.to_string())

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations ...")
    print(f"{'='*60}")
    generate_all_visualizations(csv_path)


if __name__ == "__main__":
    main()
