#!/usr/bin/env python3
"""
Visualization script for board type comparison.

Chart 1: SPaRC baseline accuracy (top) + per-board-type test accuracy (bottom).
Chart 2: Per-board-type rule-reading accuracy from object detection results
         (no baseline row).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    get_model_color,
    desaturate_color,
)

# ── Model registry ────────────────────────────────────────────────────────
# sparc CSV stem → (test / object_detection folder name, display name)
MODEL_REGISTRY = {
    "google_gemma-3-27b-it":                          ("gemma-3-27b-it",                          "Gemma 3 27B"),
    "google_gemma-4-31B-it":                          ("gemma-4-31B-it",                          "Gemma 4 31B"),
    "Qwen_Qwen3.5-27B":                              ("Qwen3.5-27B",                             "Qwen 3.5 27B"),
    "QuantTrio_Qwen3.5-397B-A17B-AWQ":               ("Qwen3.5-397B-A17B-AWQ",                  "Qwen 3.5 397B"),
    "meta-llama_Llama-4-Scout-17B-16E-Instruct":     ("Llama-4-Scout-17B-16E-Instruct",         "Llama 4 Scout"),
    "mistralai_Mistral-Small-3.2-24B-Instruct-2506": ("Mistral-Small-3.2-24B-Instruct-2506",    "Mistral Small 3.2"),
    "zai-org_GLM-4.6V":                              ("GLM-4.6V",                                "GLM 4.6V"),
}

# Fixed display order (best avg improvement → worst)
BOARD_TYPES = [
    "text",
    "path_cell_annotated",
    "coordinate_grid_and_start_end_marked",
    "start_end_marked",
    "coordinate_grid",
    "original",
]

BOARD_LABELS = {
    "original":                              "Original",
    "text":                                  "Text Symbols",
    "coordinate_grid":                       "Axis Labels",
    "start_end_marked":                      "S/E Markers",
    "coordinate_grid_and_start_end_marked":  "Axis Labels + S/E",
    "path_cell_annotated":                   "Cell Coordinates",
}


# ── Data helpers ──────────────────────────────────────────────────────────

def get_sparc_accuracy(stats_file):
    """Return overall accuracy (%) from a sparc *_vlm_stats.csv file."""
    df = pd.read_csv(stats_file)
    for _, row in df.iterrows():
        if row["Metric"] == "Correctly Solved":
            return float(str(row["Percentage"]).replace("%", ""))
    return 0.0


def _read_json_metric(model_dir, board_type, key, scale=100.0):
    """Read a numeric field from the latest *_stats_overall.json for a board type.

    Returns np.nan when no valid result is found.
    """
    pattern = f"{board_type}-B_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return np.nan

    stats_file = matches[-1]
    with open(stats_file) as f:
        data = json.load(f)

    if "error" in data or key not in data:
        return np.nan

    return data[key] * scale


def collect_data(sparc_dir, results_dir, metric_keys="accuracy", metric_scale=100.0):
    """Collect SPaRC baselines and per-board metric values for every model.

    Args:
        sparc_dir:    Path to sparc CSV directory.
        results_dir:  Path containing ``all/<model>/`` with JSON stats.
        metric_keys:  A single JSON key (str) or a list of keys to extract.
        metric_scale: Multiplier to convert raw values to percent.

    Returns:
        list of dicts with keys: display_name, sparc_acc, board_accs.
        When *metric_keys* is a string, ``board_accs[bt]`` is a float.
        When it is a list, ``board_accs[bt]`` is a dict ``{key: float}``.
    """
    multi = isinstance(metric_keys, (list, tuple))
    keys = metric_keys if multi else [metric_keys]

    results = []

    for sparc_stem, (test_folder, display_name) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        if not sparc_file.exists():
            continue

        sparc_acc = get_sparc_accuracy(sparc_file)

        model_dir = results_dir / "all" / test_folder
        board_accs = {}
        for bt in BOARD_TYPES:
            if multi:
                board_accs[bt] = {
                    k: _read_json_metric(model_dir, bt, k, metric_scale)
                    for k in keys
                }
            else:
                board_accs[bt] = _read_json_metric(model_dir, bt, keys[0], metric_scale)

        results.append({
            "display_name": display_name,
            "sparc_acc": sparc_acc,
            "board_accs": board_accs,
        })

    results.sort(key=lambda d: d["sparc_acc"])
    return results


# ── Chart builders ────────────────────────────────────────────────────────

def _shared_xlim(data):
    """Compute a shared x-axis upper limit from all accuracy values."""
    vals = [d["sparc_acc"] for d in data]
    for d in data:
        for a in d["board_accs"].values():
            if isinstance(a, dict):
                for v in a.values():
                    if not np.isnan(v):
                        vals.append(v)
            elif not np.isnan(a):
                vals.append(a)
    return max(vals) * 1.35 if vals else 50


BAR_HEIGHT = 0.55


def _draw_board_rows(axes_row, data, board_list, xlim):
    """Draw the horizontal-bar rows for each model × board type."""
    n_boards = len(board_list)

    for col, d in enumerate(data):
        ax = axes_row[col]
        color = get_model_color(d["display_name"])

        accs = [d["board_accs"].get(bt, np.nan) for bt in board_list]

        y_pos = np.arange(n_boards)
        bar_colors = [
            color if not np.isnan(a) else desaturate_color(color, 0.3)
            for a in accs
        ]
        bar_vals = [0 if np.isnan(a) else a for a in accs]
        bars = ax.barh(y_pos, bar_vals, color=bar_colors, height=BAR_HEIGHT)

        for bar, acc in zip(bars, accs):
            if np.isnan(acc):
                continue
            width = bar.get_width()
            ax.text(width + xlim * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1f}", ha="left", va="center",
                    color="black", fontweight="bold", fontsize=7)

        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.5, n_boards - 0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [BOARD_LABELS[bt] for bt in board_list] if col == 0 else []
        )
        if col == 0:
            ax.set_ylabel("Board Type", fontweight="bold")
        ax.set_xlabel("Accuracy (\\%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def create_board_comparison_chart(sparc_dir, test_dir, output_path=None):
    """Chart 1: SPaRC baseline + per-board test accuracy."""
    setup_plot_style(use_latex=True)
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.labelsize"] = 8

    data = collect_data(sparc_dir, test_dir, "accuracy", 100.0)
    if not data:
        print("No model data found!")
        return None

    n_models = len(data)
    n_boards = len(BOARD_TYPES)
    xlim = _shared_xlim(data)

    row_height = 0.20
    fig_height = (1 + n_boards) * row_height + 0.30
    fig_width = TEXT_WIDTH_INCHES

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, n_models, wspace=0.25)
    axes = np.array([fig.add_subplot(gs[0, j]) for j in range(n_models)])

    # Baseline at top, board types reversed (worst→best going down)
    boards_reversed = list(reversed(BOARD_TYPES))
    all_labels = ["Baseline"] + [BOARD_LABELS[bt] for bt in boards_reversed]
    n_rows = 1 + n_boards

    for col, d in enumerate(data):
        ax = axes[col]
        ax.set_title(d["display_name"])
        color = get_model_color(d["display_name"])

        sparc_acc = d["sparc_acc"]
        board_vals = [d["board_accs"].get(bt, np.nan) for bt in boards_reversed]

        all_vals = [sparc_acc] + board_vals
        y_pos = np.arange(n_rows)

        bar_colors = [color] + [
            color if not np.isnan(v) else desaturate_color(color, 0.3)
            for v in board_vals
        ]
        bar_data = [v if not np.isnan(v) else 0 for v in all_vals]
        bars = ax.barh(y_pos, bar_data, color=bar_colors, height=BAR_HEIGHT)

        for bar, val in zip(bars, all_vals):
            if np.isnan(val):
                continue
            ax.text(bar.get_width() + xlim * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", ha="left", va="center",
                    color="black", fontweight="bold", fontsize=7)

        # Dashed separator between baseline and board rows
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")

        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_labels if col == 0 else [])
        ax.set_xlabel("Accuracy (\\%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def create_rule_accuracy_chart(sparc_dir, od_dir, output_path=None):
    """Chart 2: Per-board rule-reading accuracy (stacked: full + average)."""
    setup_plot_style(use_latex=True)
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.labelsize"] = 8

    data = collect_data(
        sparc_dir, od_dir,
        metric_keys=["accuracy", "fraction_average"],
        metric_scale=100.0,
    )
    if not data:
        print("No object-detection data found!")
        return None

    n_models = len(data)
    n_boards = len(BOARD_TYPES)
    xlim = _shared_xlim(data)

    row_height = 0.20
    fig_height = n_boards * row_height + 0.40
    fig_width = TEXT_WIDTH_INCHES

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, n_models, wspace=0.25)
    axes = np.array([fig.add_subplot(gs[0, j]) for j in range(n_models)])

    legend_handles = []

    for col, d in enumerate(data):
        ax = axes[col]
        ax.set_title(d["display_name"])
        color = get_model_color(d["display_name"])

        y_pos = np.arange(n_boards)

        full_accs = []
        avg_accs = []
        for bt in BOARD_TYPES:
            m = d["board_accs"][bt]
            full_accs.append(m["accuracy"] if not np.isnan(m["accuracy"]) else 0)
            avg_accs.append(m["fraction_average"] if not np.isnan(m["fraction_average"]) else 0)

        remainders = [a - f for a, f in zip(avg_accs, full_accs)]

        # Avg-only portion: same color but with a diagonal hatch pattern
        bars_avg = ax.barh(y_pos, remainders, left=full_accs,
                           color=color, alpha=0.25, height=BAR_HEIGHT,
                           hatch="//", edgecolor=color, linewidth=0.5)
        # Full accuracy: solid color
        bars_full = ax.barh(y_pos, full_accs, color=color, height=BAR_HEIGHT)

        if col == 0:
            legend_handles = [bars_full, bars_avg]

        for i, a_acc in enumerate(avg_accs):
            ax.text(a_acc + xlim * 0.015, y_pos[i],
                    f"{a_acc:.1f}",
                    ha="left", va="center", color="black",
                    fontweight="bold", fontsize=7)

        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.5, n_boards - 0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [BOARD_LABELS[bt] for bt in BOARD_TYPES] if col == 0 else []
        )
        if col == 0:
            ax.set_ylabel("Board Type", fontweight="bold")
        ax.set_xlabel("Accuracy (\\%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            ["All Rules Correct", "Avg. Rule Accuracy"],
            loc="upper center", ncol=2, frameon=False,
            bbox_to_anchor=(0.5, -0.05),
            edgecolor="0.8", fancybox=False,
        )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    sparc_dir = base / "sparc"
    test_dir = base / "test"
    od_dir = base / "object_detection" / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_board_comparison_chart(
        sparc_dir, test_dir,
        output_dir / "board_comparison.pdf",
    )
    create_rule_accuracy_chart(
        sparc_dir, od_dir,
        output_dir / "rule_accuracy_comparison.pdf",
    )


if __name__ == "__main__":
    main()
