#!/usr/bin/env python3
"""
Annotated heatmap: model x board-type accuracy overview.

Rows = models (sorted by mean improvement over original, best at bottom).
Columns = SPaRC baseline + 6 board types (worst->best average).
Each cell shows the accuracy value; colour intensity encodes magnitude.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import (
    setup_plot_style, TEXT_WIDTH_INCHES, get_model_color,
    get_model_imagebox, figure_fraction_anchor_from_display_xy,
    MODEL_REGISTRY, BOARD_TYPES, BOARD_LABELS, DEFAULT_PROMPT,
    get_sparc_accuracy, read_json_metric, add_model_logos,
)


def collect_data(sparc_dir, test_dir):
    rows = []
    for sparc_stem, (test_folder, display_name) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        sparc_acc = get_sparc_accuracy(sparc_file) if sparc_file.exists() else np.nan

        model_dir = test_dir / "all" / test_folder
        board_accs = {bt: read_json_metric(model_dir, bt, "accuracy") for bt in BOARD_TYPES}

        orig = board_accs.get("original", np.nan)
        improvements = [v - orig for v in board_accs.values()
                        if not np.isnan(v) and not np.isnan(orig)]
        mean_improvement = np.mean(improvements) if improvements else 0.0

        rows.append({
            "display_name": display_name,
            "sparc": sparc_acc,
            "board_accs": board_accs,
            "_sort_key": mean_improvement,
        })

    rows.sort(key=lambda r: r["_sort_key"])
    return rows


def create_heatmap_chart(sparc_dir, test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(sparc_dir, test_dir)
    if not data:
        print("No data found!")
        return None

    columns = ["sparc"] + BOARD_TYPES
    col_labels = ["SPaRC\nBaseline"] + [BOARD_LABELS[bt] for bt in BOARD_TYPES]
    row_labels = [d["display_name"] for d in data]
    n_rows = len(row_labels)
    n_cols = len(columns)

    matrix = np.full((n_rows, n_cols), np.nan)
    for ri, d in enumerate(data):
        matrix[ri, 0] = d["sparc"]
        for ci, bt in enumerate(BOARD_TYPES):
            matrix[ri, ci + 1] = d["board_accs"][bt]

    cell_h = 0.38
    fig_w = TEXT_WIDTH_INCHES
    fig_h = n_rows * cell_h + 0.9

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    vmax = np.nanmax(matrix) * 1.02 if np.any(~np.isnan(matrix)) else 50

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "accuracy",
        ["#FFF8E7", "#FDDCB5", "#F4A261", "#E76F51", "#9B2226"],
    )
    cmap.set_bad(color="#F5F5F5")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=7, color="#BBBBBB")
            else:
                text_color = "white" if val > vmax * 0.45 else "#333333"
                txt = ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                              fontsize=7, fontweight="bold", color=text_color)
                txt.set_path_effects([
                    pe.withStroke(linewidth=2,
                                 foreground="white" if text_color == "#333333" else "#333333",
                                 alpha=0.3)
                ])

    ax.axvline(0.5, color="#444444", linewidth=1.0, linestyle="--", zorder=5)

    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i, d in enumerate(data):
        color = get_model_color(d["display_name"])
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6.5, ha="center")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True,
                   left=True, right=False, length=0)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.03, shrink=0.85)
    cbar.ax.set_ylabel("Accuracy (\\%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)

    add_model_logos(fig, ax, [d["display_name"] for d in data])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    sparc_dir = base / "sparc"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_heatmap_chart(sparc_dir, test_dir, output_dir / "heatmap_overview.pdf")


if __name__ == "__main__":
    main()
