#!/usr/bin/env python3
"""
Grouped bar chart showing the impact of worsening strategies
(low contrast, low resolution, rotated) on Qwen 3.5 397B and Gemma 4 31B,
including their path-cell-annotated recovery variants.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, get_model_color

# ── Constants ─────────────────────────────────────────────────────────────

MODELS = [
    ("Qwen3.5-397B-A17B-AWQ", "Qwen 3.5 397B"),
    ("gemma-4-31B-it",         "Gemma 4 31B"),
]

BOARD_ORDER = [
    "original",
    "path_cell_annotated",
    "low_contrast",
    "low_contrast_and_path_cell_annotated",
    "low_resolution",
    "low_resolution_and_path_cell_annotated",
    "rotated",
    "rotated_and_path_cell_annotated",
]

BOARD_LABELS = {
    "original":                              "Original",
    "path_cell_annotated":                   "Cell\nAnnotated",
    "low_contrast":                          "Low\nContrast",
    "low_contrast_and_path_cell_annotated":  "Low Contrast\n+ Annotated",
    "low_resolution":                        "Low\nResolution",
    "low_resolution_and_path_cell_annotated": "Low Res.\n+ Annotated",
    "rotated":                               "Rotated",
    "rotated_and_path_cell_annotated":       "Rotated\n+ Annotated",
}

PROMPT_TYPE = "prompt_engineering"


# ── Data loading ──────────────────────────────────────────────────────────

def _read_accuracy(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if "error" in data or "accuracy" not in data:
        return None
    return data["accuracy"] * 100.0


def collect_data(test_dir):
    """Return dict  model_folder -> list[float|nan] per board in BOARD_ORDER."""
    result = {}
    for folder, _ in MODELS:
        model_dir = test_dir / "all" / folder
        vals = []
        for bt in BOARD_ORDER:
            pattern = f"{bt}-B_{PROMPT_TYPE}-P_*_stats_overall.json"
            matches = sorted(model_dir.glob(pattern))
            if matches:
                val = _read_accuracy(matches[-1])
                vals.append(val if val is not None else np.nan)
            else:
                vals.append(np.nan)
        result[folder] = vals
    return result


# ── Chart ─────────────────────────────────────────────────────────────────

def create_worsening_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)

    n_boards = len(BOARD_ORDER)
    n_models = len(MODELS)
    bar_width = 0.35
    x = np.arange(n_boards)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.4))

    for i, (folder, display_name) in enumerate(MODELS):
        vals = data[folder]
        color = get_model_color(display_name)
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            offsets,
            [v if not np.isnan(v) else 0 for v in vals],
            width=bar_width,
            color=color,
            label=display_name,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold")

    # Separators between worsening strategy groups
    for sep_x in [1.5, 3.5, 5.5]:
        ax.axvline(sep_x, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([BOARD_LABELS[bt] for bt in BOARD_ORDER], fontsize=7)
    ax.set_ylabel("Accuracy (\\%)")

    all_vals = [v for vals in data.values() for v in vals if not np.isnan(v)]
    ax.set_ylim(0, max(all_vals) * 1.18 if all_vals else 50)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        edgecolor="0.85",
        fancybox=False,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")

    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_worsening_chart(test_dir, output_dir / "worsening_comparison.pdf")


if __name__ == "__main__":
    main()
