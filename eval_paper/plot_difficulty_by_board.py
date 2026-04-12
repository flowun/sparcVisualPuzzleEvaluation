#!/usr/bin/env python3
"""
Small-multiples line chart: accuracy vs difficulty level (1-5),
one panel per board type, lines = models.

Complements plot_difficulty.py (which shows per-model, single board type)
by showing how each representation affects the difficulty curve.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, get_model_color

# -- Constants -------------------------------------------------------------

MODEL_REGISTRY = {
    "gemma-3-27b-it":                          "Gemma 3 27B",
    "gemma-4-31B-it":                          "Gemma 4 31B",
    "Qwen3.5-27B":                             "Qwen 3.5 27B",
    "Qwen3.5-397B-A17B-AWQ":                   "Qwen 3.5 397B",
    "Llama-4-Scout-17B-16E-Instruct":          "Llama 4 Scout",
    "Mistral-Small-3.2-24B-Instruct-2506":     "Mistral Small 3.2",
    "GLM-4.6V":                                "GLM 4.6V",
}

BOARD_TYPES = [
    "original",
    "coordinate_grid",
    "start_end_marked",
    "coordinate_grid_and_start_end_marked",
    "path_cell_annotated",
    "text",
]

BOARD_LABELS = {
    "original":                              "Original",
    "coordinate_grid":                       "Axis Labels",
    "start_end_marked":                      "S/E Markers",
    "coordinate_grid_and_start_end_marked":  "Axis Labels + S/E",
    "path_cell_annotated":                   "Cell Coordinates",
    "text":                                  "Text Symbols",
}

DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


# -- Data loading ----------------------------------------------------------

def _read_difficulty_accs(model_dir, board_type):
    pattern = f"{board_type}-B_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data or "avg_accuracy_by_difficulty_level" not in data:
        return None
    abd = data["avg_accuracy_by_difficulty_level"]
    return [abd.get(str(lvl), np.nan) * 100.0
            if abd.get(str(lvl)) is not None else np.nan
            for lvl in DIFFICULTY_LEVELS]


def collect_data(test_dir):
    """Return {board_type: {display_name: [acc_per_difficulty]}}."""
    result = {}
    for bt in BOARD_TYPES:
        bt_data = {}
        for folder, display in MODEL_REGISTRY.items():
            model_dir = test_dir / "all" / folder
            accs = _read_difficulty_accs(model_dir, bt)
            if accs is not None:
                bt_data[display] = accs
        result[bt] = bt_data
    return result


# -- Chart -----------------------------------------------------------------

def create_difficulty_board_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)

    n_boards = len(BOARD_TYPES)
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.52),
        sharey=True, sharex=True,
    )

    x = np.array(DIFFICULTY_LEVELS)

    # Sort models by overall performance for consistent ordering
    model_order = sorted(MODEL_REGISTRY.values(),
                         key=lambda m: np.nanmean(
                             data["text"].get(m, [0])),
                         reverse=True)

    for idx, bt in enumerate(BOARD_TYPES):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_title(BOARD_LABELS[bt], fontsize=7, pad=3)

        for mi, model in enumerate(model_order):
            if model not in data[bt]:
                continue
            accs = data[bt][model]
            color = get_model_color(model)
            marker = MARKERS[mi % len(MARKERS)]
            ax.plot(x, accs, color=color, marker=marker,
                    markersize=5, linewidth=1.5, label=model)

        ax.set_xticks(DIFFICULTY_LEVELS)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # Shared axis labels
    for r in range(n_rows):
        axes[r][0].set_ylabel("Accuracy (\\%)", fontsize=7)
    for c in range(n_cols):
        axes[-1][c].set_xlabel("Difficulty", fontsize=7)

    # Shared legend below
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=6.5,
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )

    plt.subplots_adjust(hspace=0.35, wspace=0.08)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_difficulty_board_chart(test_dir, output_dir / "difficulty_by_board.pdf")


if __name__ == "__main__":
    main()
