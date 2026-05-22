#!/usr/bin/env python3
"""
Side-by-side confusion matrices: Original board vs. Text board.

Shows how the text representation almost eliminates object detection
errors that plague the original board (e.g. Start->Walkable confusion).
Uses Qwen 3.5 397B results only for a clean single-model comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from plot_config import setup_plot_style, COLUMN_WIDTH_INCHES

MODEL_FOLDERS = [
    "Qwen3.5-397B-A17B-AWQ",
]

ALL_MODEL_FOLDERS = [
    "gemma-3-27b-it",
    "gemma-4-31B-it",
    "Qwen3.5-27B",
    "Qwen3.5-397B-A17B-AWQ",
    "Llama-4-Scout-17B-16E-Instruct",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "GLM-4.6V",
]

ORDERED_TYPES = ["S", "E", "+", "N", "G", ".", "o", "*", "T", "P", "Y"]

TYPE_LABELS = {
    "S": "Start", "E": "End", "+": "Path", "N": "Empty", "G": "Gap",
    ".": "Dot", "o": "Square", "*": "Star", "T": "Triangle",
    "P": "Polyshape", "Y": "Neg. Poly",
}


def _normalize_cell(cell_value, for_prediction=False):
    if cell_value is None:
        return "Missing" if for_prediction else None
    if not isinstance(cell_value, str) or not cell_value:
        return "Unknown" if for_prediction else None
    abbr = cell_value[0]
    if abbr in {"A", "B", "C", "D"}:
        return "T"
    if abbr in {"S", "E", "+", "N", "G", ".", "o", "*", "T", "P", "Y"}:
        return abbr
    return "Unknown" if for_prediction else None


def _load_puzzles(model_dir, board_type):
    pattern = f"{board_type}-B_*_stats_individual.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return []
    with open(matches[-1]) as f:
        data = json.load(f)
    return list(data.values()) if isinstance(data, dict) else data


def compute_confusion_matrix(puzzles):
    raw = {t: {p: 0 for p in ORDERED_TYPES} for t in ORDERED_TYPES}

    for puzzle in puzzles:
        target = puzzle.get("puzzle_array", [])
        prediction = puzzle.get("model_solution", [])

        for ri, target_row in enumerate(target):
            for ci, target_cell in enumerate(target_row):
                true_label = _normalize_cell(target_cell, for_prediction=False)
                if true_label is None or true_label not in raw:
                    continue

                pred_cell = None
                if ri < len(prediction) and ci < len(prediction[ri]):
                    pred_cell = prediction[ri][ci]
                pred_label = _normalize_cell(pred_cell, for_prediction=True)

                if pred_label in raw[true_label]:
                    raw[true_label][pred_label] += 1

    matrix = np.zeros((len(ORDERED_TYPES), len(ORDERED_TYPES)))
    for i, t in enumerate(ORDERED_TYPES):
        for j, p in enumerate(ORDERED_TYPES):
            matrix[i, j] = raw[t][p]

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix),
                       where=row_sums > 0)
    return matrix


def collect_averaged_matrix(od_dir, board_type, folders=None):
    if folders is None:
        folders = MODEL_FOLDERS
    matrices = []
    for folder in folders:
        model_dir = od_dir / "all" / folder
        if not model_dir.exists():
            continue
        puzzles = _load_puzzles(model_dir, board_type)
        if puzzles:
            matrices.append(compute_confusion_matrix(puzzles))

    if not matrices:
        return None
    return np.mean(matrices, axis=0)


CMAP = mcolors.LinearSegmentedColormap.from_list(
    "matrix_rocket",
    [
        (0.00, "#FCFAF7"),
        (0.10, "#FBE3D2"),
        (0.25, "#F2B292"),
        (0.45, "#DE6F66"),
        (0.65, "#A93B5C"),
        (0.85, "#561A3F"),
        (1.00, "#180720"),
    ],
)

LABEL_COLOR  = "#222222"
TICK_COLOR   = "#444444"
LIGHT_TEXT   = "#FFFFFF"
DARK_TEXT    = "#2F2F2F"
SOFT_TEXT    = "#5A5A5A"


def _draw_matrix(ax, matrix, title, show_xticks=True):
    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=1.0, aspect="auto",
                   interpolation="nearest")

    n = len(ORDERED_TYPES)
    labels = [TYPE_LABELS[t] for t in ORDERED_TYPES]

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val < 0.005:
                continue
            if val >= 0.50:
                color = LIGHT_TEXT
            elif val >= 0.05:
                color = DARK_TEXT
            else:
                color = SOFT_TEXT
            fontweight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color, fontweight=fontweight)

    # Subtle white cell separators for a clean tiled look.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, color=TICK_COLOR)

    ax.set_xticks(range(n))
    if show_xticks:
        ax.set_xticklabels(labels, fontsize=6.5, rotation=40, ha="right",
                           color=TICK_COLOR)
        ax.set_xlabel("Detected Type", fontsize=8, labelpad=2,
                      color=LABEL_COLOR)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.5, color=TICK_COLOR)
    ax.set_ylabel("True Type", fontsize=8, labelpad=2, color=LABEL_COLOR)

    ax.set_title(title, fontsize=9.5, pad=2, color="#111111",
                 fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


def create_confusion_comparison(od_dir, output_path=None, folders=None):
    setup_plot_style(use_latex=True)

    matrix_orig = collect_averaged_matrix(od_dir, "original", folders=folders)
    matrix_text = collect_averaged_matrix(od_dir, "text", folders=folders)

    if matrix_orig is None or matrix_text is None:
        print("Missing data for confusion matrices!")
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 1.25),
        gridspec_kw={"hspace": 0.15},
    )

    _draw_matrix(ax1, matrix_orig, "Original Board", show_xticks=False)
    im = _draw_matrix(ax2, matrix_text, "Text Symbols", show_xticks=True)

    cbar = fig.colorbar(
        im, ax=[ax1, ax2], orientation="vertical",
        fraction=0.038, pad=0.025, shrink=0.95, aspect=30,
    )
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.5,
                        color=TICK_COLOR, pad=2)
    for label in cbar.ax.get_yticklabels():
        label.set_color(TICK_COLOR)
    cbar.set_label("Row Frequency", fontsize=7.5, labelpad=6,
                   color=LABEL_COLOR)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("#666666")
    cbar.outline.set_linewidth(0.6)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    od_dir = base / "object_detection" / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    # Single-model view (Qwen 3.5 397B).
    create_confusion_comparison(
        od_dir,
        output_dir / "od_confusion_comparison.pdf",
        folders=MODEL_FOLDERS,
    )
    # Averaged across all 7 core models.
    create_confusion_comparison(
        od_dir,
        output_dir / "od_confusion_comparison_all_models.pdf",
        folders=ALL_MODEL_FOLDERS,
    )


if __name__ == "__main__":
    main()
