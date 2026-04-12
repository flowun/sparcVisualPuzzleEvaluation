#!/usr/bin/env python3
"""
Grouped horizontal bar chart comparing SPaRC baseline accuracy
(original evaluation protocol) vs. controlled evaluation accuracy
(original board, prompt_engineering) for all 7 models.

Shows that the SPaRC baseline is systematically higher because it uses
a different prompt (with textual coordinates) and evaluation pipeline.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    MODEL_REGISTRY, DEFAULT_PROMPT, VARIANT_COLORS,
    get_sparc_accuracy, read_json_metric, add_model_logos,
)


def collect_data(sparc_dir, test_dir):
    results = []
    for sparc_stem, (test_folder, display_name) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        sparc_acc = get_sparc_accuracy(sparc_file) if sparc_file.exists() else np.nan

        model_dir = test_dir / "all" / test_folder
        ctrl_acc = read_json_metric(model_dir, "original", "accuracy",
                                    prompt_type=DEFAULT_PROMPT)

        if not np.isnan(sparc_acc):
            results.append((display_name, sparc_acc, ctrl_acc))

    results.sort(key=lambda r: r[1])
    return results


def create_sparc_vs_controlled_chart(sparc_dir, test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(sparc_dir, test_dir)
    if not data:
        print("No data found!")
        return None

    n = len(data)
    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.7))

    sparc_vals = [d[1] for d in data]
    ctrl_vals = [d[2] if not np.isnan(d[2]) else 0 for d in data]

    bars_sparc = ax.barh(y + bar_h / 2, sparc_vals, height=bar_h,
                         color=VARIANT_COLORS["sparc"], alpha=0.85,
                         label="SPaRC Baseline",
                         edgecolor="white", linewidth=0.5)
    bars_ctrl = ax.barh(y - bar_h / 2, ctrl_vals, height=bar_h,
                        color=VARIANT_COLORS["traceback"], alpha=0.85,
                        label="Controlled (Original)",
                        edgecolor="white", linewidth=0.5)

    xlim = max(sparc_vals) * 1.45

    for i, (name, sparc_v, ctrl_v) in enumerate(data):
        ctrl_v_safe = ctrl_v if not np.isnan(ctrl_v) else 0
        delta = ctrl_v_safe - sparc_v

        if sparc_v > 0.5:
            txt = ax.text(sparc_v + xlim * 0.01,
                          i + bar_h / 2,
                          f"{sparc_v:.1f}", ha="left", va="center",
                          fontsize=6, fontweight="bold",
                          color=VARIANT_COLORS["sparc"])
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        if ctrl_v_safe > 0.5:
            txt = ax.text(ctrl_v_safe + xlim * 0.01,
                          i - bar_h / 2,
                          f"{ctrl_v_safe:.1f}", ha="left", va="center",
                          fontsize=6, fontweight="bold",
                          color=VARIANT_COLORS["traceback"])
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        sign = "+" if delta >= 0 else ""
        delta_txt = ax.text(max(sparc_v, ctrl_v_safe) + xlim * 0.08,
                            i,
                            f"{sign}{delta:.1f}pp", ha="left", va="center",
                            fontsize=5.5, color="#555555")
        delta_txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xlim(0, xlim)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=7)
    for i, (name, _, _) in enumerate(data):
        ax.get_yticklabels()[i].set_color(get_model_color(name))
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Accuracy (\\%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(fontsize=6, loc="lower right", frameon=True,
              framealpha=0.95, fancybox=False)

    add_model_logos(fig, ax, [d[0] for d in data])

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

    create_sparc_vs_controlled_chart(
        sparc_dir, test_dir, output_dir / "sparc_vs_controlled.pdf")


if __name__ == "__main__":
    main()
