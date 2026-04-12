#!/usr/bin/env python3
"""
Line chart comparing prompt types (default, default + text repr,
prompt engineering) across board types for Qwen 3.5 397B.

Shows the impact of prompt design on task accuracy, with SPaRC baseline
as a horizontal reference line.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, BOARD_TYPES, BOARD_LABELS,
    BOARD_LABELS_MULTILINE,
    get_sparc_accuracy, read_json_metric,
)

PROMPT_TYPES = ["default_no_tr", "default_tr", "prompt_engineering"]

PROMPT_LABELS = {
    "default_no_tr":      "Default",
    "default_tr":         "Default + Text Repr.",
    "prompt_engineering":  "Prompt Engineering",
}

PROMPT_COLORS = {
    "default_no_tr":      "#8DA0CB",
    "default_tr":         "#66C2A5",
    "prompt_engineering":  "#FC8D62",
}

PROMPT_MARKERS = {
    "default_no_tr":      "s",
    "default_tr":         "D",
    "prompt_engineering":  "o",
}

MODEL_FOLDER = "Qwen3.5-397B-A17B-AWQ"
SPARC_CSV_STEM = "QuantTrio_Qwen3.5-397B-A17B-AWQ"

BOARD_ORDER = list(reversed(BOARD_TYPES))


def collect_data(test_dir):
    """For each prompt_type x board_type, read accuracy for Qwen 3.5 397B."""
    model_dir = test_dir / "all" / MODEL_FOLDER
    if not model_dir.exists():
        return {}

    result = {}
    for pt in PROMPT_TYPES:
        vals = []
        for bt in BOARD_ORDER:
            val = read_json_metric(model_dir, bt, "accuracy", prompt_type=pt)
            vals.append(val)
        result[pt] = vals
    return result


def create_prompt_comparison_chart(test_dir, sparc_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    sparc_file = sparc_dir / f"{SPARC_CSV_STEM}_vlm_stats.csv"
    sparc_acc = get_sparc_accuracy(sparc_file) if sparc_file.exists() else None

    n_boards = len(BOARD_ORDER)
    x = np.arange(n_boards)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.8))

    for pt in PROMPT_TYPES:
        vals = data[pt]
        ax.plot(x, vals,
                color=PROMPT_COLORS[pt],
                marker=PROMPT_MARKERS[pt],
                markersize=5,
                linewidth=1.5,
                label=PROMPT_LABELS[pt],
                zorder=3)

        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 1.5, f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=5.5, fontweight="bold",
                        color=PROMPT_COLORS[pt])

    if sparc_acc is not None:
        ax.axhline(sparc_acc, color="#333333", linewidth=1.0, linestyle="--",
                   zorder=2, alpha=0.7)
        ax.text(n_boards - 0.7, sparc_acc + 1.2,
                f"SPaRC Baseline ({sparc_acc:.1f}\\%)",
                fontsize=6, color="#333333", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([BOARD_LABELS_MULTILINE[bt] for bt in BOARD_ORDER], fontsize=7)
    ax.set_ylabel("Accuracy (\\%)")

    all_valid = [v for vals in data.values() for v in vals if not np.isnan(v)]
    y_top = max(max(all_valid), sparc_acc or 0) * 1.15
    ax.set_ylim(0, y_top)
    ax.set_xlim(-0.3, n_boards - 0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    ax.legend(
        fontsize=7,
        loc="upper left",
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
    sparc_dir = base / "sparc"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_prompt_comparison_chart(test_dir, sparc_dir, output_dir / "prompt_comparison.pdf")


if __name__ == "__main__":
    main()
