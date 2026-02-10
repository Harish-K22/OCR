"""
Worker script: loads ONE model, runs it on all dataset categories, outputs JSON results.
Called by run_evaluation.py as a subprocess to isolate memory usage.

Usage:
    python _run_single_model.py '{"module":"models.easy_ocr","cls":"EasyOCRModel","kwargs":{}}'
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from evaluation.metrics import compute_cer, compute_wer, compute_accuracy


def get_dataset_pairs(images_dir: Path, gt_dir: Path) -> list[tuple[Path, str]]:
    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        gt_path = gt_dir / f"{img_path.stem}.txt"
        if gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            pairs.append((img_path, gt_text))
    return pairs


def evaluate(model, dataset, category):
    rows = []
    for img_path, gt_text in dataset:
        start = time.perf_counter()
        try:
            prediction = model.extract_text(str(img_path))
        except Exception as e:
            print(f"INFO: [FAIL] {img_path.name}: {e}", flush=True)
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
        print(f"INFO: {model.get_name()} | {category} | {img_path.name} | "
              f"CER={rows[-1]['cer']:.3f} | {elapsed:.1f}s", flush=True)
    return rows


def main():
    spec = json.loads(sys.argv[1])
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["cls"])
    model = cls(**spec.get("kwargs", {}))

    print(f"INFO: Loading {model.get_name()} ...", flush=True)
    model.load_model()
    print(f"INFO: {model.get_name()} loaded.", flush=True)

    results = []

    # Run on ALL categories dynamically
    for cat_key in config.CATEGORIES:
        img_dir, gt_dir = config.get_category_dirs(cat_key)
        pairs = get_dataset_pairs(img_dir, gt_dir)
        if pairs:
            print(f"INFO: Running on {cat_key} ({len(pairs)} images) ...", flush=True)
            results.extend(evaluate(model, pairs, cat_key))

    # Output results as JSON on a special line the parent process reads
    print(f"RESULT:{json.dumps(results)}", flush=True)


if __name__ == "__main__":
    main()
