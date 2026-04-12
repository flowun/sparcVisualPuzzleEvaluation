#!/usr/bin/env python3
"""
Grouped bar chart showing accuracy by board-size bin across
key representations.  Averaged across all models.

Board sizes are binned by grid area (width x height).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import setup_plot_style, COLUMN_WIDTH_INCHES

# -- Constants -------------------------------------------------------------

MODEL_FOLDERS = [
    "gemma-3-27b-it",
    "gemma-4-31B-it",
    "Qwen3.5-27B",
    "Qwen3.5-397B-A17B-AWQ",
    "Llama-4-Scout-17B-16E-Instruct",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "GLM-4.6V",
]

BOARD_TYPES = [
    "original",
    "coordinate_grid_and_start_end_marked",
    "path_cell_annotated",
    "text",
]

BOARD_LABELS = {
    "original":                              "Original",
    "coordinate_grid_and_start_end_marked":  "Axis Labels + S/E",
    "path_cell_annotated":                   "Cell Coordinates",
    "text":                                  "Text Symbols",
}

BOARD_COLORS = {
    "original":                              "#E53935",
    "coordinate_grid_and_start_end_marked":  "#FB8C00",
    "path_cell_annotated":                   "#8E24AA",
    "text":                                  "#00897B",
}

SIZE_BINS = [
    (0, 35, r"$\leq$35"),
    (36, 63, "36--63"),
    (64, 99, "64--99"),
    (100, 200, r"$\geq$100"),
]


# -- Data helpers ----------------------------------------------------------

def _load_individual(model_dir, board_type):
    pattern = f"{board_type}-B_*_stats_individual.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return []
    with open(matches[-1]) as f:
        data = json.load(f)
    return list(data.values()) if isinstance(data, dict) else data


def collect_data(test_dir):
    """Return {board_type: {bin_label: [accuracy values across models]}}."""
    result = {bt: {lbl: [] for _, _, lbl in SIZE_BINS} for bt in BOARD_TYPES}

    for folder in MODEL_FOLDERS:
        model_dir = test_dir / "all" / folder
        for bt in BOARD_TYPES:
            puzzles = _load_individual(model_dir, bt)
            if not puzzles:
                continue

            # Group puzzles by size bin
            bin_correct = {lbl: [] for _, _, lbl in SIZE_BINS}
            for puzzle in puzzles:
                w = puzzle.get("width", 0)
                h = puzzle.get("height", 0)
                area = w * h
                valid = 1.0 if puzzle.get("is_valid", False) else 0.0

                for lo, hi, lbl in SIZE_BINS:
                    if lo <= area <= hi:
                        bin_correct[lbl].append(valid)
                        break

            # Per-model accuracy for each bin
            for _, _, lbl in SIZE_BINS:
                vals = bin_correct[lbl]
                if vals:
                    result[bt][lbl].append(np.mean(vals) * 100.0)

    # Average across models
    return {
        bt: {lbl: np.mean(vals) if vals else 0.0
             for lbl, vals in bt_data.items()}
        for bt, bt_data in result.items()
    }


# -- Chart -----------------------------------------------------------------

def create_size_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    bin_labels = [lbl for _, _, lbl in SIZE_BINS]
    n_bins = len(bin_labels)
    n_boards = len(BOARD_TYPES)
    bar_w = 0.20  # slightly wider bars
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.75))

    for bi, bt in enumerate(BOARD_TYPES):
        vals = [data[bt][lbl] for lbl in bin_labels]
        offset = (bi - (n_boards - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w,
                      color=BOARD_COLORS[bt], label=BOARD_LABELS[bt],
                      edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, vals):
            if v > 1:
                txt = ax.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.5,
                              f"{v:.0f}", ha="center", va="bottom",
                              fontsize=5, fontweight="bold",
                              color=BOARD_COLORS[bt])
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_xlabel("Board Area (width $\\times$ height)")
    ax.set_ylabel("Accuracy (\\%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(fontsize=6, loc="upper right", frameon=True,
              framealpha=0.95, fancybox=False,
              handletextpad=0.4, columnspacing=0.6)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_size_chart(test_dir, output_dir / "board_size_accuracy.pdf")


if __name__ == "__main__":
    main()
