#!/usr/bin/env python3
"""
Line chart of accuracy vs puzzle difficulty (1-5), comparing the original
SPaRC evaluation (dashed) with our controlled evaluation on the text board
(solid). One pair of lines per model, all on a single panel.
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
    df = pd.read_csv(stats_file)
    accs = [np.nan] * 5
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip()
        for lvl in DIFFICULTY_LEVELS:
            if metric == f"Difficulty {lvl} Solved":
                accs[lvl - 1] = float(str(row["Percentage"]).replace("%", ""))
    return accs


def get_test_difficulty(model_dir):
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
        results.append({"display_name": display_name,
                        "sparc_accs": sparc_accs,
                        "test_accs": test_accs})
    results.sort(key=lambda d: np.nanmean(d["test_accs"]), reverse=True)
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

    ax.set_xticks(DIFFICULTY_LEVELS)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel(r"Accuracy (\%)")
    ax.set_ylim(-2, 102)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([], [], color="0.4", linewidth=1.3, linestyle="--", alpha=0.5,
               label="Original SPaRC"),
        Line2D([], [], color="0.4", linewidth=1.3, label="Text Symbols (Controlled)"),
    ]
    fig.legend(
        style_handles, [h.get_label() for h in style_handles],
        loc="upper center", bbox_to_anchor=(0.5, -0.04),
        fontsize=7, ncol=2, frameon=False,
        columnspacing=1.4, handlelength=2.0,
    )

    model_handles, model_labels = ax.get_legend_handles_labels()
    rows = [(model_handles[:3], model_labels[:3]),
            (model_handles[3:5], model_labels[3:5]),
            (model_handles[5:7], model_labels[5:7])]
    for i, (h, l) in enumerate(rows):
        if not h:
            continue
        fig.legend(
            h, l,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10 - i * 0.06),
            fontsize=7, ncol=len(h), frameon=False,
            handletextpad=0.4, columnspacing=1.4,
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

    create_difficulty_chart(sparc_dir, test_dir,
                            output_dir / "difficulty_comparison.pdf")


if __name__ == "__main__":
    main()
