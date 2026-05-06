#!/usr/bin/env python3
"""
Horizontal dumbbell chart: per-model improvement from the original board
to the best-performing board representation.

Each row shows a line from original accuracy (open dot) to best accuracy
(filled dot), coloured by model.  The best board type is annotated.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.transforms import blended_transform_factory
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    MODEL_REGISTRY, BOARD_TYPES, BOARD_LABELS_SHORT, DEFAULT_PROMPT,
    read_json_metric, get_model_imagebox,
)


def collect_data(test_dir):
    """Return list of (display_name, original_acc, best_acc, best_board)."""
    results = []
    for _, (test_folder, display_name) in MODEL_REGISTRY.items():
        model_dir = test_dir / "all" / test_folder
        orig = read_json_metric(model_dir, "original", "accuracy",
                                prompt_type=DEFAULT_PROMPT)
        if np.isnan(orig):
            continue

        best_acc, best_board = orig, "original"
        for bt in BOARD_TYPES:
            val = read_json_metric(model_dir, bt, "accuracy",
                                   prompt_type=DEFAULT_PROMPT)
            if not np.isnan(val) and val > best_acc:
                best_acc, best_board = val, bt

        results.append((display_name, orig, best_acc, best_board))

    results.sort(key=lambda r: r[2] - r[1])
    return results


def create_delta_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    n = len(data)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.85))

    for i, (name, orig, best, best_bt) in enumerate(data):
        color = get_model_color(name)
        delta = best - orig

        ax.plot([orig, best], [i, i], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=2)

        ax.scatter(orig, i, color="white", edgecolors=color, linewidths=1.5,
                   s=50, zorder=3)

        ax.scatter(best, i, color=color, edgecolors=color, linewidths=0.5,
                   s=50, zorder=3)

        sign = "+" if delta >= 0 else ""
        label = f"{sign}{delta:.1f}pp"
        if best_bt != "original":
            label += f" ({BOARD_LABELS_SHORT[best_bt]})"
        ax.text(best + 2.5, i, label, ha="left", va="center",
                fontsize=8, color=color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=8, color="black")

    ax.set_xlabel("Accuracy (\\%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    all_bests = [d[2] for d in data]
    ax.set_xlim(-1, max(all_bests) * 1.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([], [], marker="o", color="gray", markerfacecolor="white",
               markeredgecolor="gray", markeredgewidth=1.2, markersize=5,
               linestyle="", label="Original"),
        Line2D([], [], marker="o", color="gray", markerfacecolor="gray",
               markeredgecolor="gray", markersize=5,
               linestyle="", label="Best representation"),
    ]
    ax.legend(handles=leg_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.95, fancybox=False,
              handletextpad=0.4)

    # Place each model logo to the left of its y-tick label. Use the actual
    # tick-label extent so the logo lands cleanly to the left of the longest
    # model name regardless of font / figure size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_axes = ax.transAxes.inverted()
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (name, _, _, _) in enumerate(data):
        imagebox = get_model_imagebox(name, zoom_factor=0.85)
        if imagebox is None:
            continue
        tick_label = ax.get_yticklabels()[i]
        bbox = tick_label.get_window_extent(renderer)
        axes_x_left, _ = inv_axes.transform((bbox.x0, 0))
        ab = AnnotationBbox(
            imagebox,
            (axes_x_left, i),
            xycoords=trans,
            xybox=(-3, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(1.0, 0.5),
            pad=0,
            zorder=10,
        )
        ax.add_artist(ab)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_delta_chart(test_dir, output_dir / "improvement_delta.pdf")


if __name__ == "__main__":
    main()
