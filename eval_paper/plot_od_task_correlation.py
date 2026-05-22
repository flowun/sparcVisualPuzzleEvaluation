#!/usr/bin/env python3
"""
Scatter plot with regression line showing the correlation between
full-board object-detection accuracy and task-solving accuracy,
matched by model x board type.

Uses OD 'accuracy' (fraction of puzzles where ALL rules were detected
correctly) rather than 'fraction_average' (mean per-rule accuracy),
since the former correlates much more strongly with task performance
(r=0.96 vs r=0.51): a model must read the entire board correctly
to solve the puzzle.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    MODEL_REGISTRY, BOARD_TYPES, BOARD_LABELS, BOARD_MARKERS,
    DEFAULT_PROMPT, read_json_metric,
)

MODEL_FOLDERS = {v[0]: v[1] for v in MODEL_REGISTRY.values()}


def collect_data(test_dir, od_dir, od_metric="accuracy"):
    """Return list of (display_name, board_type, od_metric, task_acc)."""
    points = []
    for folder, display in MODEL_FOLDERS.items():
        task_model_dir = test_dir / "all" / folder
        od_model_dir = od_dir / "all" / folder

        if not task_model_dir.exists() or not od_model_dir.exists():
            continue

        for bt in BOARD_TYPES:
            od_val = read_json_metric(od_model_dir, bt, od_metric,
                                      prompt_type="default")
            task_acc = read_json_metric(task_model_dir, bt, "accuracy",
                                        prompt_type=DEFAULT_PROMPT)
            if not np.isnan(od_val) and not np.isnan(task_acc):
                points.append((display, bt, od_val, task_acc))

    return points


def create_correlation_chart(test_dir, od_dir, output_path=None,
                             od_metric="accuracy",
                             xlabel="Board Detection Acc. (\\%)",
                             height_ratio=0.85):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir, od_dir, od_metric=od_metric)
    if not data:
        print("No data found!")
        return None

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * height_ratio))

    for display, bt, od_acc, task_acc in data:
        color = get_model_color(display)
        ax.scatter(od_acc, task_acc, color=color, marker=BOARD_MARKERS[bt],
                   s=40, alpha=0.8, edgecolors="white", linewidth=0.3, zorder=3)

    x_arr = np.array([d[2] for d in data])
    y_arr = np.array([d[3] for d in data])

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)
    r_squared = r_value ** 2

    x_line = np.linspace(max(x_arr.min() - 3, 0), min(x_arr.max() + 3, 100), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#333333", linewidth=1.5, linestyle="--",
            zorder=2, alpha=0.7)

    n = len(x_arr)
    x_mean = x_arr.mean()
    se = np.sqrt(np.sum((y_arr - (slope * x_arr + intercept)) ** 2) / (n - 2))
    t_val = stats.t.ppf(0.975, n - 2)
    ci = t_val * se * np.sqrt(1 / n + (x_line - x_mean) ** 2 / np.sum((x_arr - x_mean) ** 2))
    ax.fill_between(x_line, y_line - ci, y_line + ci,
                    color="#333333", alpha=0.08, zorder=1)

    p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.05, 0.95,
            f"$r = {r_value:.2f}$\n$R^2 = {r_squared:.2f}$\n${p_str}$",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.8", alpha=0.9))

    seen_models = []
    model_handles = []
    for display, _, _, _ in data:
        if display not in seen_models:
            seen_models.append(display)
            model_handles.append(
                mlines.Line2D([], [], color=get_model_color(display), marker="o",
                              linestyle="", markersize=5, label=display,
                              markeredgecolor="white", markeredgewidth=0.3)
            )

    board_handles = [
        mlines.Line2D([], [], color="#555555", marker=BOARD_MARKERS[bt],
                      linestyle="", markersize=5,
                      label=BOARD_LABELS[bt])
        for bt in BOARD_TYPES
    ]

    # Board type legend: bottom right inside the plot, single column.
    leg_board = ax.legend(handles=board_handles, fontsize=7,
                          loc="lower right",
                          bbox_to_anchor=(1.03, 0.0),
                          frameon=True, framealpha=0.95, fancybox=False,
                          title=r"\textbf{Board Type}", title_fontsize=7,
                          handletextpad=0.4, ncol=1, borderpad=0.4,
                          labelspacing=0.3)
    ax.add_artist(leg_board)

    # Model legend below the plot, 3 / 2 / 2 each row centered.
    rows = [model_handles[:3], model_handles[3:5], model_handles[5:7]]
    y_top = -0.02
    row_height = 0.055
    for i, row in enumerate(rows):
        fig.legend(handles=row, fontsize=7,
                   loc="upper center",
                   bbox_to_anchor=(0.5, y_top - i * row_height),
                   frameon=False, ncol=len(row),
                   handletextpad=0.4, columnspacing=0.9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Task Accuracy (\\%)")
    ax.set_ylim(bottom=-1)
    ax.set_xlim(left=max(x_arr.min() - 5, -1))

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
    od_dir = base / "object_detection" / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    # Full-board OD accuracy (all rules correct) vs task accuracy.
    create_correlation_chart(
        test_dir, od_dir,
        output_dir / "od_task_correlation.pdf",
        od_metric="accuracy",
        xlabel="Board Detection Acc. (\\%)",
        height_ratio=0.75,
    )
    # Average per-rule OD accuracy vs task accuracy.
    create_correlation_chart(
        test_dir, od_dir,
        output_dir / "od_task_correlation_avg.pdf",
        od_metric="fraction_average",
        xlabel="Avg. Rule Detection Acc. (\\%)",
        height_ratio=1.0,
    )


if __name__ == "__main__":
    main()
