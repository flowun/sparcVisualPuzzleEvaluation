#!/usr/bin/env python3
"""Vertical bar chart of SPaRC text-only baseline accuracy per model, with
a Human reference column. Models are sorted by accuracy (descending) and
each bar carries the model logo above it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox
from pathlib import Path

from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color, get_model_imagebox,
)
from plot_board_comparison import MODEL_REGISTRY, get_sparc_accuracy


HUMAN_SOLVE_RATE = 98.0


def collect_baseline_data(sparc_dir):
    results = []
    for sparc_stem, (_, display_name) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        if not sparc_file.exists():
            continue
        acc = get_sparc_accuracy(sparc_file)
        results.append({"display_name": display_name, "accuracy": acc})
    results.sort(key=lambda d: d["accuracy"], reverse=True)
    return results


def create_baseline_chart(sparc_dir, output_path=None):
    setup_plot_style(use_latex=True)

    model_data = collect_baseline_data(sparc_dir)
    if not model_data:
        print("No baseline data found!")
        return None

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, 1.8))

    all_display_names = ["Human"] + [m["display_name"] for m in model_data]
    all_accuracies = [HUMAN_SOLVE_RATE] + [m["accuracy"] for m in model_data]
    all_colors = ["#2E86AB"] + [get_model_color(m["display_name"]) for m in model_data]

    x_pos = np.arange(len(all_display_names))
    bars = ax.bar(x_pos, all_accuracies, color=all_colors,
                  edgecolor="white", linewidth=0.5, zorder=2)

    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1.5,
               alpha=0.7, zorder=1)

    for i, (bar, acc, color) in enumerate(zip(bars, all_accuracies, all_colors)):
        height = bar.get_height()

        if i == 0:
            ax.annotate(
                f"{acc:.0f}\\%",
                xy=(bar.get_x() + bar.get_width() / 2, height - 18),
                ha="center", va="top",
                fontsize=6, fontweight="bold", color="white",
            )
            imagebox = get_model_imagebox("Human")
            if imagebox:
                ab = AnnotationBbox(
                    imagebox,
                    (bar.get_x() + bar.get_width() / 2, height - 8),
                    xycoords="data", frameon=False, pad=0,
                )
                ax.add_artist(ab)
        else:
            ax.annotate(
                f"{acc:.1f}\\%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6, fontweight="bold", color=color,
            )
            m = model_data[i - 1]
            imagebox = get_model_imagebox(m["display_name"])
            if imagebox:
                ab = AnnotationBbox(
                    imagebox,
                    (bar.get_x() + bar.get_width() / 2, height),
                    xybox=(0, 15), xycoords="data",
                    boxcoords="offset points",
                    frameon=False, pad=0,
                )
                ax.add_artist(ab)

    ax.set_ylabel("Accuracy (\\%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_display_names, rotation=45, ha="right")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlim(-0.5, len(all_display_names) - 0.5)
    ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    sparc_dir = base / "sparc"
    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)
    create_baseline_chart(sparc_dir, output_dir / "baseline_comparison.pdf")


if __name__ == "__main__":
    main()
