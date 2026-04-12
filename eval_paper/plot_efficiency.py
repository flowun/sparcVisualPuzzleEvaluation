#!/usr/bin/env python3
"""
Scatter plot: average completion tokens vs. accuracy per board type
for all 7 models.  Board-type markers + model colors show whether
additional tokens translate to higher accuracy.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    BOARD_TYPES, BOARD_LABELS, BOARD_MARKERS, DEFAULT_PROMPT,
)

MODEL_REGISTRY = {
    "gemma-3-27b-it":                          "Gemma 3 27B",
    "gemma-4-31B-it":                          "Gemma 4 31B",
    "Qwen3.5-27B":                             "Qwen 3.5 27B",
    "Qwen3.5-397B-A17B-AWQ":                   "Qwen 3.5 397B",
    "Llama-4-Scout-17B-16E-Instruct":          "Llama 4 Scout",
    "Mistral-Small-3.2-24B-Instruct-2506":     "Mistral Small 3.2",
    "GLM-4.6V":                                "GLM 4.6V",
}

from plot_config import BOARD_LABELS_SHORT


def collect_data(test_dir):
    result = {}
    for folder, display in MODEL_REGISTRY.items():
        model_dir = test_dir / "all" / folder
        if not model_dir.exists():
            continue
        points = []
        for bt in BOARD_TYPES:
            pattern = f"{bt}-B_{DEFAULT_PROMPT}-P_*_stats_overall.json"
            matches = sorted(model_dir.glob(pattern))
            if not matches:
                continue
            with open(matches[-1]) as f:
                data = json.load(f)
            if "error" in data or "accuracy" not in data:
                continue
            acc = data["accuracy"] * 100.0
            tokens = data.get("avg_tokens_per_task", {}).get("completion_tokens", np.nan)
            if not np.isnan(tokens):
                points.append((bt, acc, tokens / 1000))
        if points:
            result[display] = points
    return result


def create_efficiency_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.85))

    for display, points in data.items():
        color = get_model_color(display)
        for bt, acc, tok in points:
            ax.scatter(tok, acc, color=color, marker=BOARD_MARKERS[bt],
                       s=45, edgecolors="white", linewidths=0.4, zorder=3)

    model_handles = [
        mlines.Line2D([], [], color=get_model_color(d), marker="o",
                      linestyle="", markersize=5, label=d,
                      markeredgecolor="white", markeredgewidth=0.3)
        for d in data
    ]

    board_handles = [
        mlines.Line2D([], [], color="#555555", marker=BOARD_MARKERS[bt],
                      linestyle="", markersize=5, label=BOARD_LABELS_SHORT[bt])
        for bt in BOARD_TYPES
    ]

    leg1 = ax.legend(handles=model_handles, fontsize=6, loc="upper left",
                     frameon=True, framealpha=0.95, fancybox=False,
                     title="\\textbf{Model}", title_fontsize=6,
                     handletextpad=0.4)
    ax.add_artist(leg1)
    ax.legend(handles=board_handles, fontsize=6, loc="lower right",
              frameon=True, framealpha=0.95, fancybox=False,
              title="\\textbf{Board}", title_fontsize=6,
              handletextpad=0.3, ncol=2, columnspacing=0.5)

    ax.set_xlabel("Avg. Completion Tokens (k)")
    ax.set_ylabel("Accuracy (\\%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_efficiency_chart(test_dir, output_dir / "efficiency.pdf")


if __name__ == "__main__":
    main()
