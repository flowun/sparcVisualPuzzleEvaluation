#!/usr/bin/env python3
"""
Single-panel line chart of average completion-token usage vs puzzle difficulty.

One solid line per model (text board, prompt engineering). Model legend
laid out in 3 / 2 / 2 rows below the chart — same layout as plot_difficulty.py.
"""

import json
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


def _read_tokens_by_difficulty(model_dir, board_type):
    pattern = f"{board_type}-B_{DEFAULT_PROMPT}-P_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return [np.nan] * 5
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data:
        return [np.nan] * 5
    abd = data.get("avg_tokens_per_task_by_difficulty_level")
    if not abd:
        return [np.nan] * 5
    return [abd[str(lvl)]["completion_tokens"]
            if str(lvl) in abd else np.nan
            for lvl in DIFFICULTY_LEVELS]


def collect_data(test_dir):
    results = []
    for _, (test_folder, display_name) in MODEL_REGISTRY.items():
        model_dir = test_dir / "all" / test_folder
        if not model_dir.exists():
            continue
        toks = _read_tokens_by_difficulty(model_dir, BOARD_TYPE)
        if any(not np.isnan(t) for t in toks):
            results.append({"display_name": display_name, "tokens": toks})
    results.sort(key=lambda d: np.nanmean(d["tokens"]), reverse=True)
    return results


def _add_3_2_2_legend(fig, handles, labels, *,
                      y_top=-0.04, row_height=0.06, fontsize=7):
    """Place a 3 / 2 / 2 legend below the chart, each row independently centered."""
    rows = [(handles[:3], labels[:3]),
            (handles[3:5], labels[3:5]),
            (handles[5:7], labels[5:7])]
    for i, (h, l) in enumerate(rows):
        if not h:
            continue
        fig.legend(
            h, l,
            loc="upper center",
            bbox_to_anchor=(0.5, y_top - i * row_height),
            fontsize=fontsize,
            ncol=len(h),
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.4,
        )


def create_token_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
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
        ax.plot(x, d["tokens"], color=color, marker=marker,
                markersize=4, linewidth=1.3, label=d["display_name"])

    ax.set_xticks(DIFFICULTY_LEVELS)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Completion Tokens")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    _add_3_2_2_legend(fig, handles, labels)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_token_chart(test_dir, output_dir / "token_usage.pdf")


if __name__ == "__main__":
    main()
