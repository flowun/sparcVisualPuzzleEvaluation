#!/usr/bin/env python3
"""
Radar plot of avg_path_analysis_metrics per board type (+ SPaRC baseline),
with each metric averaged across all models.

Reduced to 4 lines for readability: SPaRC baseline, Original, Cell Annotated, Text.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES,
    MODEL_REGISTRY, DEFAULT_PROMPT,
)

SELECTED_BOARDS = ["original", "path_cell_annotated", "text"]

BOARD_LABELS = {
    "original":           "Original",
    "path_cell_annotated": "Cell Coordinates",
    "text":               "Text Symbols",
}

PATH_METRICS = [
    "starts_at_start_ends_at_exit",
    "connected_line",
    "non_intersecting_line",
    "no_rule_crossing",
    "fully_valid_path",
]

METRIC_LABELS = {
    "starts_at_start_ends_at_exit": "Correct\nStart/End",
    "connected_line":               "Connected\nPath",
    "non_intersecting_line":        "Non-\nIntersecting",
    "no_rule_crossing":             "No Rule\nViolation",
    "fully_valid_path":             "Fully\nValid",
}

SPARC_CSV_METRIC_MAP = {
    "Correct Start/End":  "starts_at_start_ends_at_exit",
    "Connected Paths":    "connected_line",
    "Non-Intersecting":   "non_intersecting_line",
    "No Rule Violations": "no_rule_crossing",
    "Fully Valid Paths":  "fully_valid_path",
}

BOARD_COLORS = {
    "original":           "#B71C1C",
    "path_cell_annotated": "#E65100",
    "text":               "#6A1B9A",
}

SPARC_COLOR = "#333333"


def _get_sparc_path_metrics(stats_file):
    df = pd.read_csv(stats_file)
    metrics = {}
    for _, row in df.iterrows():
        name = str(row["Metric"]).strip()
        if name in SPARC_CSV_METRIC_MAP:
            pct = float(str(row["Percentage"]).replace("%", ""))
            metrics[SPARC_CSV_METRIC_MAP[name]] = pct / 100.0
    return metrics


def _get_test_path_metrics(model_dir, board_type):
    pattern = f"{board_type}-B_{DEFAULT_PROMPT}-P_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data or "avg_path_analysis_metrics" not in data:
        return None
    return data["avg_path_analysis_metrics"]


def collect_radar_data(sparc_dir, test_dir):
    per_board = {bt: [] for bt in SELECTED_BOARDS}
    sparc_all = []

    for sparc_stem, (test_folder, _) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        if not sparc_file.exists():
            continue

        sparc_all.append(_get_sparc_path_metrics(sparc_file))

        model_dir = test_dir / "all" / test_folder
        for bt in SELECTED_BOARDS:
            m = _get_test_path_metrics(model_dir, bt)
            if m is not None:
                per_board[bt].append(m)

    def _avg_metrics(dicts):
        out = {}
        for key in PATH_METRICS:
            vals = [d[key] for d in dicts if key in d]
            out[key] = np.mean(vals) if vals else 0.0
        return out

    board_avg = {bt: _avg_metrics(dicts) for bt, dicts in per_board.items()}
    sparc_avg = _avg_metrics(sparc_all)

    return board_avg, sparc_avg


def create_radar_chart(sparc_dir, test_dir, output_path=None):
    setup_plot_style(use_latex=True)
    plt.rcParams["axes.labelsize"] = 8

    board_avg, sparc_avg = collect_radar_data(sparc_dir, test_dir)

    n_metrics = len(PATH_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES),
        subplot_kw={"projection": "polar"},
    )

    markers = {"original": "o", "path_cell_annotated": "s", "text": "P"}

    sparc_vals = [sparc_avg.get(m, 0) for m in PATH_METRICS] + \
                 [sparc_avg.get(PATH_METRICS[0], 0)]
    ax.plot(angles, sparc_vals, linewidth=2.0, color=SPARC_COLOR,
            linestyle="--", marker="X", markersize=5,
            label="SPaRC Baseline", zorder=5)
    ax.fill(angles, sparc_vals, color=SPARC_COLOR, alpha=0.05)

    for bt in SELECTED_BOARDS:
        color = BOARD_COLORS[bt]
        vals = [board_avg[bt].get(m, 0) for m in PATH_METRICS] + \
               [board_avg[bt].get(PATH_METRICS[0], 0)]
        ax.plot(angles, vals, linewidth=1.6, color=color,
                marker=markers[bt], markersize=5,
                label=BOARD_LABELS[bt], zorder=6)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [METRIC_LABELS[m] for m in PATH_METRICS],
        fontsize=7,
    )

    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["20\\%", "40\\%", "60\\%", "80\\%", "100\\%"],
        fontsize=6, color="0.4",
    )
    ax.set_rlabel_position(180 / n_metrics)

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=7,
        frameon=False,
        ncol=4,
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

    create_radar_chart(sparc_dir, test_dir, output_dir / "path_analysis_radar.pdf")


if __name__ == "__main__":
    main()
