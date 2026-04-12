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
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES

MODEL_FOLDERS = [
    "Qwen3.5-397B-A17B-AWQ",
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


def collect_averaged_matrix(od_dir, board_type):
    matrices = []
    for folder in MODEL_FOLDERS:
        model_dir = od_dir / "all" / folder
        if not model_dir.exists():
            continue
        puzzles = _load_puzzles(model_dir, board_type)
        if puzzles:
            matrices.append(compute_confusion_matrix(puzzles))

    if not matrices:
        return None
    return np.mean(matrices, axis=0)


def _draw_matrix(ax, matrix, title, show_ylabels=True):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blues", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1.0, aspect="equal")

    n = len(ORDERED_TYPES)
    labels = [TYPE_LABELS[t] for t in ORDERED_TYPES]

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val < 0.005:
                continue
            text_color = "white" if val >= 0.45 else "black"
            fontweight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=text_color, fontweight=fontweight)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#e0e0e0", linewidth=0.4, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha="right")
    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels(labels, fontsize=5.5)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("Detected Type", fontsize=7)
    if show_ylabels:
        ax.set_ylabel("True Type", fontsize=7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


def create_confusion_comparison(od_dir, output_path=None):
    setup_plot_style(use_latex=True)

    matrix_orig = collect_averaged_matrix(od_dir, "original")
    matrix_text = collect_averaged_matrix(od_dir, "text")

    if matrix_orig is None or matrix_text is None:
        print("Missing data for confusion matrices!")
        return None

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.42),
        gridspec_kw={"wspace": 0.15, "width_ratios": [1.12, 1]},
    )

    _draw_matrix(ax1, matrix_orig, "Original Board", show_ylabels=True)
    im = _draw_matrix(ax2, matrix_text, "Text Board", show_ylabels=False)

    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.025, pad=0.03,
                        shrink=0.85)
    cbar.ax.set_ylabel("Row-Normalized Frequency", fontsize=7)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_visible(False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    od_dir = base / "object_detection" / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_confusion_comparison(od_dir, output_dir / "od_confusion_comparison.pdf")


if __name__ == "__main__":
    main()
