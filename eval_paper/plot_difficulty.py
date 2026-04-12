#!/usr/bin/env python3
"""
Line chart showing model accuracy by puzzle difficulty level (1-5).

Dashed lines = original SPaRC evaluation.
Solid lines = controlled evaluation with best board (text) + prompt_engineering.

Annotates the Qwen 3.5 scaling observation: its accuracy decreases linearly
with difficulty while other models show a more logarithmic (steep early) drop.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    MODEL_REGISTRY, DEFAULT_PROMPT,
)

DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
BOARD_TYPE = "text"


def get_sparc_difficulty(stats_file):
    """Return list of accuracy (%) for difficulty 1-5 from a sparc CSV."""
    df = pd.read_csv(stats_file)
    accs = [np.nan] * 5
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip()
        for lvl in DIFFICULTY_LEVELS:
            if metric == f"Difficulty {lvl} Solved":
                accs[lvl - 1] = float(str(row["Percentage"]).replace("%", ""))
    return accs


def get_test_difficulty(model_dir):
    """Return difficulty accuracy (%) for the text board type, levels 1-5."""
    pattern = f"{BOARD_TYPE}-B_{DEFAULT_PROMPT}-P_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return [np.nan] * 5
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data or "avg_accuracy_by_difficulty_level" not in data:
        return [np.nan] * 5
    abd = data["avg_accuracy_by_difficulty_level"]
    return [abd.get(str(lvl), np.nan) * 100.0 if abd.get(str(lvl)) is not None
            else np.nan for lvl in DIFFICULTY_LEVELS]


def collect_data(sparc_dir, test_dir):
    results = []
    for sparc_stem, (test_folder, display_name) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        if not sparc_file.exists():
            continue

        sparc_accs = get_sparc_difficulty(sparc_file)
        model_dir = test_dir / "all" / test_folder
        test_accs = get_test_difficulty(model_dir)

        results.append({
            "display_name": display_name,
            "sparc_accs": sparc_accs,
            "test_accs": test_accs,
        })

    results.sort(key=lambda d: np.nanmean(d["sparc_accs"]), reverse=True)
    return results


def create_difficulty_chart(sparc_dir, test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(sparc_dir, test_dir)
    if not data:
        print("No data found!")
        return None

    x = np.array(DIFFICULTY_LEVELS)

    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.85),
    )

    for i, d in enumerate(data):
        color = get_model_color(d["display_name"])
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(x, d["sparc_accs"], color=color, marker=marker,
                markersize=4, linewidth=1.3, linestyle="--", alpha=0.5)

        ax.plot(x, d["test_accs"], color=color, marker=marker,
                markersize=4, linewidth=1.3, label=d["display_name"])

    # Annotate the Qwen 3.5 scaling finding
    qwen35_data = [d for d in data if "Qwen 3.5 397B" in d["display_name"]]
    if qwen35_data:
        d = qwen35_data[0]
        accs = d["test_accs"]
        if not any(np.isnan(a) for a in accs) and accs[0] > 0:
            mid_x = 3
            mid_y = accs[2] if not np.isnan(accs[2]) else 30
            ax.annotate(
                "Linear decay",
                xy=(mid_x, mid_y), xytext=(mid_x + 0.6, mid_y + 12),
                fontsize=5.5, color=get_model_color(d["display_name"]),
                arrowprops=dict(arrowstyle="->", color=get_model_color(d["display_name"]),
                                lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=get_model_color(d["display_name"]), alpha=0.9,
                          linewidth=0.6),
            )

    ax.set_xticks(DIFFICULTY_LEVELS)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(-2, 102)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([], [], color="0.4", linewidth=1.3, linestyle="--", alpha=0.5,
               label="Original evaluation"),
        Line2D([], [], color="0.4", linewidth=1.3,
               label="Controlled evaluation"),
    ]

    model_handles, model_labels = ax.get_legend_handles_labels()
    all_handles = style_handles + model_handles
    all_labels = ["Original evaluation", "Controlled evaluation"] + model_labels

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=7,
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
    )

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

    create_difficulty_chart(sparc_dir, test_dir, output_dir / "difficulty_comparison.pdf")


if __name__ == "__main__":
    main()
