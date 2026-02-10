"""
Run API-based OCR model (Mistral Pixtral) on ALL dataset categories.
Run this AFTER run_evaluation.py has verified all local models work.
Results are appended to the existing CSV and visualizations regenerated.

Usage:
    MISTRAL_API_KEY=xxx python run_api_models.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from tqdm import tqdm

import config
from evaluation.metrics import compute_cer, compute_wer, compute_accuracy
from evaluation.visualize import generate_all_visualizations
from models.mistral_ocr import MistralOCR


def get_dataset_pairs(images_dir: Path, gt_dir: Path) -> list[tuple[Path, str]]:
    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        gt_path = gt_dir / f"{img_path.stem}.txt"
        if gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            pairs.append((img_path, gt_text))
    return pairs


def evaluate_model(
    model, dataset: list[tuple[Path, str]], category: str, desc: str = "",
) -> list[dict]:
    """Run one model on one dataset, with rate-limit-friendly delays."""
    rows = []
    for img_path, gt_text in tqdm(dataset, desc=desc, leave=False):
        start = time.perf_counter()
        try:
            prediction = model.extract_text(str(img_path))
        except Exception as e:
            print(f"    [FAIL] Error on {img_path.name}: {e}")
            prediction = ""
        elapsed = time.perf_counter() - start

        rows.append({
            "model": model.get_name(),
            "category": category,
            "image": img_path.name,
            "cer": round(compute_cer(prediction, gt_text), 4),
            "wer": round(compute_wer(prediction, gt_text), 4),
            "accuracy": round(compute_accuracy(prediction, gt_text), 2),
            "time_sec": round(elapsed, 3),
            "prediction": prediction[:200],
            "ground_truth": gt_text[:200],
        })
        print(f"    {model.get_name()} | {category} | {img_path.name} | "
              f"CER={rows[-1]['cer']:.3f} | {elapsed:.1f}s")

        # Rate limiting: Mistral free tier allows ~2 requests/minute
        time.sleep(30.0)

    return rows


def main() -> None:
    csv_path = config.SCORES_DIR / "all_results.csv"

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        print(f"Loaded {len(existing_df)} existing results from {csv_path}")
    else:
        existing_df = pd.DataFrame()
        print("No existing results found. Starting fresh.")

    # Load all categories
    all_datasets = {}
    print(f"\n{'='*60}")
    print("DATASET CATEGORIES")
    print(f"{'='*60}")
    for cat_key in config.CATEGORIES:
        img_dir, gt_dir = config.get_category_dirs(cat_key)
        pairs = get_dataset_pairs(img_dir, gt_dir)
        if pairs:
            all_datasets[cat_key] = pairs
            label = config.get_category_label(cat_key)
            print(f"  {label}: {len(pairs)} images")

    if not all_datasets:
        print("\nNo datasets found. Run prepare_dataset.py first.")
        sys.exit(1)

    total_images = sum(len(p) for p in all_datasets.values())
    print(f"\nTotal: {total_images} images across {len(all_datasets)} categories")

    # Load Mistral
    try:
        mistral = MistralOCR()
        mistral.load_model()
        print("[OK] Mistral OCR ready")
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Set MISTRAL_API_KEY in your .env file.")
        sys.exit(1)

    print(f"\nRunning Mistral OCR on {total_images} images ...")
    print(f"Estimated time: ~{total_images * 30 // 60} minutes (30s delay per request)")

    new_results = []

    print(f"\n{'='*60}")
    print(f"Running Mistral OCR ...")
    print(f"{'='*60}")

    for cat_key, pairs in all_datasets.items():
        label = config.get_category_label(cat_key)
        print(f"  Mistral OCR on {label} ({len(pairs)} images) ...")
        new_results.extend(
            evaluate_model(mistral, pairs, cat_key, desc=f"    Mistral [{cat_key}]")
        )

    # Merge and save
    new_df = pd.DataFrame(new_results)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["model", "category", "image"], keep="last"
    )
    combined.to_csv(csv_path, index=False)
    print(f"\nCombined results saved to {csv_path} ({len(combined)} rows)")

    # Print summary
    print(f"\n{'='*60}")
    print("MISTRAL OCR SUMMARY")
    print(f"{'='*60}")
    api_summary = new_df.groupby(["model", "category"]).agg(
        avg_cer=("cer", "mean"),
        avg_wer=("wer", "mean"),
        avg_acc=("accuracy", "mean"),
        avg_time=("time_sec", "mean"),
    ).round(3)
    print(api_summary.to_string())

    # Regenerate visualizations with all data
    print(f"\n{'='*60}")
    print("Regenerating visualizations with all models ...")
    print(f"{'='*60}")
    generate_all_visualizations(csv_path)


if __name__ == "__main__":
    main()
