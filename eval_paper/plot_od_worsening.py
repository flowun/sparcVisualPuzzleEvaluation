#!/usr/bin/env python3
"""
Grouped bar chart showing object-detection accuracy (fraction_average)
under worsening conditions and their annotation-recovery variants.

Demonstrates that cell annotations recover object detection accuracy
under degraded visual conditions. Uses Gemma 4 31B and Qwen 3.5 397B.

Includes group labels and recovery delta annotations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import (
    setup_plot_style, TEXT_WIDTH_INCHES, get_model_color,
    read_json_metric,
)

MODELS = [
    ("Qwen3.5-397B-A17B-AWQ", "Qwen 3.5 397B"),
    ("gemma-4-31B-it",         "Gemma 4 31B"),
]

BOARD_ORDER = [
    "original",
    "path_cell_annotated",
    "low_contrast",
    "low_contrast_and_path_cell_annotated",
    "low_resolution",
    "low_resolution_and_path_cell_annotated",
    "rotated",
    "rotated_and_path_cell_annotated",
]

BOARD_LABELS = {
    "original":                              "Original",
    "path_cell_annotated":                   "Cell\nCoordinates",
    "low_contrast":                          "Low\nContrast",
    "low_contrast_and_path_cell_annotated":  "Low Contrast\n+ Cell Coord.",
    "low_resolution":                        "Low\nResolution",
    "low_resolution_and_path_cell_annotated": "Low Res.\n+ Cell Coord.",
    "rotated":                               "Rotated",
    "rotated_and_path_cell_annotated":       "Rotated\n+ Cell Coord.",
}

GROUP_LABELS = {
    1.5: "Baseline",
    3.5: "Low Contrast",
    5.5: "Low Resolution",
    7.0: "Rotation",
}

RECOVERY_PAIRS = [(2, 3), (4, 5), (6, 7)]


def collect_data(od_dir):
    result = {}
    for folder, _ in MODELS:
        model_dir = od_dir / "all" / folder
        vals = []
        for bt in BOARD_ORDER:
            val = read_json_metric(model_dir, bt, "fraction_average",
                                   prompt_type="default")
            vals.append(val)
        result[folder] = vals
    return result


def create_od_worsening_chart(od_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(od_dir)

    n_boards = len(BOARD_ORDER)
    n_models = len(MODELS)
    bar_width = 0.35
    x = np.arange(n_boards)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.45))

    all_vals_flat = []

    for i, (folder, display_name) in enumerate(MODELS):
        vals = data[folder]
        color = get_model_color(display_name)
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            offsets,
            [v if not np.isnan(v) else 0 for v in vals],
            width=bar_width,
            color=color,
            label=display_name,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            all_vals_flat.append(v)
            txt = ax.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.3,
                          f"{v:.1f}",
                          ha="center", va="bottom",
                          fontsize=5.5, fontweight="bold",
                          color=color)
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    y_max = max(all_vals_flat) * 1.18 if all_vals_flat else 100

    for sep_x in [1.5, 3.5, 5.5]:
        ax.axvline(sep_x, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)

    for gx, label in GROUP_LABELS.items():
        ax.text(gx, y_max * 0.98, label,
                ha="center", va="top", fontsize=6, color="#555555",
                fontstyle="italic")

    for worsened_idx, recovered_idx in RECOVERY_PAIRS:
        for mi, (folder, display_name) in enumerate(MODELS):
            vals = data[folder]
            w_val = vals[worsened_idx]
            r_val = vals[recovered_idx]
            if np.isnan(w_val) or np.isnan(r_val):
                continue
            delta = r_val - w_val
            mid_x = (worsened_idx + recovered_idx) / 2 + (mi - (n_models - 1) / 2) * bar_width
            color = get_model_color(display_name)
            sign = "+" if delta >= 0 else ""
            txt = ax.text(mid_x, max(w_val, r_val) + y_max * 0.04,
                          f"{sign}{delta:.1f}",
                          ha="center", va="bottom", fontsize=5,
                          color=color, fontweight="bold")
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([BOARD_LABELS[bt] for bt in BOARD_ORDER], fontsize=6.5)
    ax.set_ylabel("Avg. Rule Detection Acc. (\\%)")
    ax.set_ylim(0, y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fancybox=False,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")

    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    od_dir = base / "object_detection" / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_od_worsening_chart(od_dir, output_dir / "od_worsening_comparison.pdf")


if __name__ == "__main__":
    main()
