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
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    MODEL_REGISTRY, BOARD_TYPES, BOARD_LABELS_SHORT, DEFAULT_PROMPT,
    read_json_metric, add_model_logos,
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

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.75))

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
        txt = ax.text(best + 1.2, i, label, ha="left", va="center",
                      fontsize=5.5, color=color, fontweight="bold")
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=7)
    for i, (name, _, _, _) in enumerate(data):
        ax.get_yticklabels()[i].set_color(get_model_color(name))
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Accuracy (\\%)")

    all_bests = [d[2] for d in data]
    ax.set_xlim(-1, max(all_bests) * 1.45)

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
    ax.legend(handles=leg_handles, fontsize=6, loc="lower right",
              frameon=True, framealpha=0.95, fancybox=False,
              handletextpad=0.4)

    add_model_logos(fig, ax, [d[0] for d in data])

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
