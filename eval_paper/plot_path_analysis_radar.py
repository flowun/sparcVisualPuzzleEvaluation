#!/usr/bin/env python3
"""
Radars of avg_path_analysis_metrics across all six controlled board representations.
Emits two separate figures: averaged over all models, and Qwen 3.5 397B alone.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES,
    MODEL_REGISTRY, DEFAULT_PROMPT,
    BOARD_TYPES, BOARD_LABELS, BOARD_COLORS, BOARD_MARKERS,
)

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

# (a)-(f) display order; reverses BOARD_TYPES (which is most-to-least aided).
BOARD_ORDER = list(reversed(BOARD_TYPES))

QWEN_LARGE_FOLDER = "Qwen3.5-397B-A17B-AWQ"


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


def _collect_for_models(test_dir, model_folders):
    per_board = {bt: [] for bt in BOARD_TYPES}
    for test_folder in model_folders:
        model_dir = test_dir / "all" / test_folder
        for bt in BOARD_TYPES:
            m = _get_test_path_metrics(model_dir, bt)
            if m is not None:
                per_board[bt].append(m)

    out = {}
    for bt, dicts in per_board.items():
        out[bt] = {}
        for key in PATH_METRICS:
            vals = [d[key] for d in dicts if key in d]
            out[bt][key] = float(np.mean(vals)) if vals else 0.0
    return out


def _build_figure(board_avg, output_path):
    setup_plot_style(use_latex=True)
    plt.rcParams["axes.labelsize"] = 8

    angles = np.linspace(0, 2 * np.pi, len(PATH_METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.8),
        subplot_kw={"projection": "polar"},
    )
    fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.18)

    for bt in BOARD_ORDER:
        vals = [board_avg[bt].get(m, 0) for m in PATH_METRICS] + \
               [board_avg[bt].get(PATH_METRICS[0], 0)]
        ax.plot(angles, vals, linewidth=1.6, color=BOARD_COLORS[bt],
                marker=BOARD_MARKERS[bt], markersize=5,
                label=BOARD_LABELS[bt], zorder=6)
        ax.fill(angles, vals, color=BOARD_COLORS[bt], alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in PATH_METRICS], fontsize=7)

    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["20\\%", "40\\%", "60\\%", "80\\%", "100\\%"],
        fontsize=6, color="0.4",
    )
    ax.set_rlabel_position(180 / len(PATH_METRICS))

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        fontsize=7,
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handlelength=2.0,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close(fig)


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    all_folders = [test_folder for _, (test_folder, _) in MODEL_REGISTRY.items()]
    avg_data = _collect_for_models(test_dir, all_folders)
    qwen_data = _collect_for_models(test_dir, [QWEN_LARGE_FOLDER])

    _build_figure(avg_data, output_dir / "path_analysis_radar.pdf")
    _build_figure(qwen_data, output_dir / "path_analysis_radar_qwen.pdf")


if __name__ == "__main__":
    main()
