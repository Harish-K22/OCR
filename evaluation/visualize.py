"""
Generate all comparison charts and save them to results/visualizations/.

Charts produced (multi-category):
    1. CER per category (one chart per category)
    2. Accuracy per category (one chart per category)
    3. Overall CER heatmap: models × categories
    4. Overall Accuracy heatmap: models × categories
    5. Best model per category bar chart
    6. WER grouped comparison
    7. Processing time comparison
    8. Radar / spider chart (overall)
    9. Summary table as image
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

matplotlib.use("Agg")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})

VIS_DIR = config.VIS_DIR


def _save(fig, name: str) -> None:
    path = VIS_DIR / f"{name}.png"
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


def _get_colors(n: int):
    return sns.color_palette("husl", n)


# ═════════════════════════════════════════════════════════════
# 1. CER bar chart for EACH category
# ═════════════════════════════════════════════════════════════

def plot_cer_per_category(df: pd.DataFrame) -> None:
    categories = df["category"].unique()
    for i, cat in enumerate(sorted(categories)):
        subset = df[df["category"] == cat]
        means = subset.groupby("model")["cer"].mean().sort_values()
        colors = _get_colors(len(means))

        fig, ax = plt.subplots(figsize=(10, max(4, len(means) * 0.7)))
        bars = ax.barh(means.index, means.values, color=colors)
        ax.set_xlabel("Character Error Rate (CER) — lower is better")
        label = config.CATEGORIES.get(cat, (cat, cat))[1] if cat in config.CATEGORIES else cat
        ax.set_title(f"CER: {label}")
        ax.set_xlim(0, min(max(means.values) * 1.2, 1.05))
        for bar, val in zip(bars, means.values):
            ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        _save(fig, f"01_cer_{cat}")


# ═════════════════════════════════════════════════════════════
# 2. Accuracy bar chart for EACH category
# ═════════════════════════════════════════════════════════════

def plot_accuracy_per_category(df: pd.DataFrame) -> None:
    categories = df["category"].unique()
    for i, cat in enumerate(sorted(categories)):
        subset = df[df["category"] == cat]
        means = subset.groupby("model")["accuracy"].mean().sort_values(ascending=True)
        colors = _get_colors(len(means))

        fig, ax = plt.subplots(figsize=(10, max(4, len(means) * 0.7)))
        bars = ax.barh(means.index, means.values, color=colors)
        ax.set_xlabel("Accuracy (%) — higher is better")
        label = config.CATEGORIES.get(cat, (cat, cat))[1] if cat in config.CATEGORIES else cat
        ax.set_title(f"Accuracy: {label}")
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, means.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        _save(fig, f"02_accuracy_{cat}")


# ═════════════════════════════════════════════════════════════
# 3. CER Heatmap: models × categories
# ═════════════════════════════════════════════════════════════

def plot_cer_heatmap(df: pd.DataFrame) -> None:
    pivot = df.groupby(["model", "category"])["cer"].mean().unstack()
    # Rename columns to display labels
    col_map = {}
    for col in pivot.columns:
        if col in config.CATEGORIES:
            col_map[col] = config.CATEGORIES[col][1]
        else:
            col_map[col] = col
    pivot = pivot.rename(columns=col_map)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2), max(5, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title("CER Heatmap: Models × Categories (lower is better)", fontsize=13)
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    _save(fig, "03_cer_heatmap")


# ═════════════════════════════════════════════════════════════
# 4. Accuracy Heatmap: models × categories
# ═════════════════════════════════════════════════════════════

def plot_accuracy_heatmap(df: pd.DataFrame) -> None:
    pivot = df.groupby(["model", "category"])["accuracy"].mean().unstack()
    col_map = {}
    for col in pivot.columns:
        if col in config.CATEGORIES:
            col_map[col] = config.CATEGORIES[col][1]
        else:
            col_map[col] = col
    pivot = pivot.rename(columns=col_map)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2), max(5, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, vmin=0, vmax=100)
    ax.set_title("Accuracy Heatmap: Models × Categories (higher is better)", fontsize=13)
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    _save(fig, "04_accuracy_heatmap")


# ═════════════════════════════════════════════════════════════
# 5. Best model per category
# ═════════════════════════════════════════════════════════════

def plot_best_per_category(df: pd.DataFrame) -> None:
    pivot = df.groupby(["model", "category"])["accuracy"].mean().unstack()
    best_models = pivot.idxmax()
    best_scores = pivot.max()

    categories = sorted(best_models.index)
    labels = []
    for cat in categories:
        if cat in config.CATEGORIES:
            labels.append(config.CATEGORIES[cat][1])
        else:
            labels.append(cat)
    models = [best_models[cat] for cat in categories]
    scores = [best_scores[cat] for cat in categories]

    # Unique model colors
    unique_models = list(set(models))
    palette = dict(zip(unique_models, _get_colors(len(unique_models))))
    colors = [palette[m] for m in models]

    fig, ax = plt.subplots(figsize=(12, max(5, len(categories) * 0.8)))
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlabel("Best Accuracy (%)")
    ax.set_title("Best Performing Model per Category")
    ax.set_xlim(0, 105)
    for bar, score, model in zip(bars, scores, models):
        ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{model} ({score:.1f}%)", va="center", fontsize=9)
    _save(fig, "05_best_per_category")


# ═════════════════════════════════════════════════════════════
# 6. WER grouped bar comparison
# ═════════════════════════════════════════════════════════════

def plot_wer(df: pd.DataFrame) -> None:
    pivot = df.groupby(["model", "category"])["wer"].mean().unstack()
    col_map = {}
    for col in pivot.columns:
        if col in config.CATEGORIES:
            col_map[col] = config.CATEGORIES[col][1]
        else:
            col_map[col] = col
    pivot = pivot.rename(columns=col_map)

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Word Error Rate (WER) — lower is better")
    ax.set_title("WER Across All Categories")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "06_wer_comparison")


# ═════════════════════════════════════════════════════════════
# 7. Processing time comparison
# ═════════════════════════════════════════════════════════════

def plot_time(df: pd.DataFrame) -> None:
    means = df.groupby("model")["time_sec"].mean().sort_values()
    colors = _get_colors(len(means))

    fig, ax = plt.subplots(figsize=(10, max(4, len(means) * 0.7)))
    bars = ax.barh(means.index, means.values, color=colors)
    ax.set_xlabel("Average Time per Image (seconds)")
    ax.set_title("Processing Speed Comparison")
    for bar, val in zip(bars, means.values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", va="center", fontsize=9)
    _save(fig, "07_processing_time")


# ═════════════════════════════════════════════════════════════
# 8. Radar / Spider chart
# ═════════════════════════════════════════════════════════════

def plot_radar(df: pd.DataFrame) -> None:
    # One axis per category (accuracy)
    pivot = df.groupby(["model", "category"])["accuracy"].mean().unstack()
    models = pivot.index.tolist()
    categories = sorted(pivot.columns.tolist())
    labels = []
    for cat in categories:
        if cat in config.CATEGORIES:
            labels.append(config.CATEGORIES[cat][1].split("(")[0].strip())
        else:
            labels.append(cat)

    N_axes = len(labels)
    if N_axes < 3:
        return  # need at least 3 axes for radar

    angles = [n / float(N_axes) * 2 * math.pi for n in range(N_axes)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    colors = _get_colors(len(models))
    for i, model in enumerate(models):
        values = [pivot.loc[model, cat] if cat in pivot.columns else 0 for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.08, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy Radar: Which Model Excels Where?", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    _save(fig, "08_radar_chart")


# ═════════════════════════════════════════════════════════════
# 9. Summary table
# ═════════════════════════════════════════════════════════════

def plot_summary_table(df: pd.DataFrame) -> None:
    summary = df.groupby("model").agg(
        Avg_CER=("cer", "mean"),
        Avg_WER=("wer", "mean"),
        Avg_Accuracy=("accuracy", "mean"),
        Avg_Time=("time_sec", "mean"),
    ).round(3)
    summary = summary.sort_values("Avg_CER")

    fig, ax = plt.subplots(figsize=(12, 3 + 0.4 * len(summary)))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    ax.set_title("Overall OCR Model Performance Summary", fontsize=13, pad=20)
    _save(fig, "09_summary_table")


# ═════════════════════════════════════════════════════════════
# 10. Per-category summary table (which model wins where)
# ═════════════════════════════════════════════════════════════

def plot_category_winner_table(df: pd.DataFrame) -> None:
    pivot = df.groupby(["model", "category"])["accuracy"].mean().unstack()
    best = pivot.idxmax()
    best_score = pivot.max()

    rows = []
    for cat in sorted(pivot.columns):
        label = config.CATEGORIES.get(cat, (cat, cat))[1] if cat in config.CATEGORIES else cat
        rows.append([label, best[cat], f"{best_score[cat]:.1f}%"])

    fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Category", "Best Model", "Accuracy"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.5)
    ax.set_title("Best Model for Each Use Case", fontsize=14, pad=20)
    _save(fig, "10_category_winners")


# ═════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════

def generate_all_visualizations(csv_path: str | Path | None = None) -> None:
    """Load results CSV and produce all charts."""
    if csv_path is None:
        csv_path = config.SCORES_DIR / "all_results.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} result rows from {csv_path}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print()

    plot_cer_per_category(df)
    plot_accuracy_per_category(df)
    plot_cer_heatmap(df)
    plot_accuracy_heatmap(df)
    plot_best_per_category(df)
    plot_wer(df)
    plot_time(df)
    plot_radar(df)
    plot_summary_table(df)
    plot_category_winner_table(df)

    print(f"\nAll visualizations saved to {VIS_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()
