#!/usr/bin/env python3
"""
Improvement heatmap: models (rows) x board types (columns).

Each cell shows the percentage-point improvement over the original board
representation.  Highlights which vision aid benefits which model most.
The 'original' column is omitted (it is the baseline with delta = 0).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import (
    setup_plot_style, TEXT_WIDTH_INCHES, get_model_color,
    MODEL_REGISTRY, BOARD_TYPES, BOARD_LABELS, DEFAULT_PROMPT,
    read_json_metric, add_model_logos,
)

IMPROVEMENT_BOARDS = [bt for bt in BOARD_TYPES if bt != "original"]


def collect_data(test_dir):
    rows = []
    for sparc_stem, (test_folder, display_name) in MODEL_REGISTRY.items():
        model_dir = test_dir / "all" / test_folder
        orig = read_json_metric(model_dir, "original", "accuracy",
                                prompt_type=DEFAULT_PROMPT)

        deltas = {}
        for bt in IMPROVEMENT_BOARDS:
            val = read_json_metric(model_dir, bt, "accuracy",
                                   prompt_type=DEFAULT_PROMPT)
            if not np.isnan(val) and not np.isnan(orig):
                deltas[bt] = val - orig
            else:
                deltas[bt] = np.nan

        vals = [v for v in deltas.values() if not np.isnan(v)]
        mean_delta = np.mean(vals) if vals else 0.0

        rows.append({
            "display_name": display_name,
            "orig": orig,
            "deltas": deltas,
            "_sort_key": mean_delta,
        })

    rows.sort(key=lambda r: r["_sort_key"])
    return rows


def create_improvement_heatmap(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    col_labels = [BOARD_LABELS[bt] for bt in IMPROVEMENT_BOARDS]
    row_labels = [d["display_name"] for d in data]
    n_rows = len(row_labels)
    n_cols = len(IMPROVEMENT_BOARDS)

    matrix = np.full((n_rows, n_cols), np.nan)
    for ri, d in enumerate(data):
        for ci, bt in enumerate(IMPROVEMENT_BOARDS):
            matrix[ri, ci] = d["deltas"][bt]

    cell_h = 0.42
    fig_w = TEXT_WIDTH_INCHES
    fig_h = n_rows * cell_h + 0.9

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanmax(np.abs(matrix)) * 1.05 if np.any(~np.isnan(matrix)) else 20

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "delta",
        ["#2166AC", "#92C5DE", "#F7F7F7", "#FDDBC7", "#D6604D", "#B2182B"],
    )
    cmap.set_bad(color="#F0F0F0")

    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=7, color="#BBBBBB")
            else:
                sign = "+" if val >= 0 else ""
                text_color = "white" if abs(val) > vmax * 0.55 else "#333333"
                txt = ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                              fontsize=7, fontweight="bold", color=text_color)
                txt.set_path_effects([
                    pe.withStroke(linewidth=2,
                                 foreground="white" if text_color == "#333333" else "#333333",
                                 alpha=0.3)
                ])

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
    ax.set_xticklabels(col_labels, fontsize=7, ha="center")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True,
                   left=True, right=False, length=0)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, shrink=0.85)
    cbar.ax.set_ylabel("Improvement over Original (pp)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)

    add_model_logos(fig, ax, [d["display_name"] for d in data])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_improvement_heatmap(test_dir, output_dir / "improvement_heatmap.pdf")


if __name__ == "__main__":
    main()
