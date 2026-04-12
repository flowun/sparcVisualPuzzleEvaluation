#!/usr/bin/env python3
"""
Line chart showing per-rule-type detection accuracy across board types,
averaged over all models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES

# ── Constants ─────────────────────────────────────────────────────────────

MODEL_FOLDERS = [
    "gemma-3-27b-it",
    "gemma-4-31B-it",
    "Qwen3.5-27B",
    "Qwen3.5-397B-A17B-AWQ",
    "Llama-4-Scout-17B-16E-Instruct",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "GLM-4.6V",
]

BOARD_ORDER = [
    "original",
    "coordinate_grid",
    "start_end_marked",
    "coordinate_grid_and_start_end_marked",
    "path_cell_annotated",
    "text",
]

BOARD_LABELS = {
    "original":                              "Original",
    "coordinate_grid":                       "Axis\nLabels",
    "start_end_marked":                      "S/E\nMarkers",
    "coordinate_grid_and_start_end_marked":  "Axis Labels\n+ S/E",
    "path_cell_annotated":                   "Cell\nCoordinates",
    "text":                                  "Text\nSymbols",
}

RULE_TYPES = ["+", "N", ".", "o", "*", "T", "G", "S", "E", "P", "Y"]

RULE_LABELS = {
    "+": "Walkable",
    "N": "Empty Rule",
    ".": "Dot",
    "o": "Square",
    "*": "Star",
    "T": "Triangle",
    "G": "Gap",
    "S": "Start",
    "E": "End",
    "P": "Polyshape",
    "Y": "Neg. Polyshape",
}

RULE_COLORS = {
    "+": "#66C2A5",
    "N": "#8DA0CB",
    ".": "#333333",
    "o": "#E78AC3",
    "*": "#FFD92F",
    "T": "#FC8D62",
    "G": "#A6D854",
    "S": "#1B9E77",
    "E": "#D95F02",
    "P": "#7570B3",
    "Y": "#E7298A",
}

RULE_MARKERS = {
    "+": "o",
    "N": "s",
    ".": ".",
    "o": "D",
    "*": "*",
    "T": "^",
    "G": "v",
    "S": ">",
    "E": "<",
    "P": "P",
    "Y": "X",
}


# ── Data loading ──────────────────────────────────────────────────────────

def _load_accuracy_by_type(model_dir, board_type):
    """Return accuracy_by_type dict from the latest stats JSON, or None."""
    pattern = f"{board_type}-B_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data or "accuracy_by_type" not in data:
        return None
    return data["accuracy_by_type"]


def collect_data(od_dir):
    """Build a matrix: rule_type × board_type, averaged across models.

    Returns:
        dict  rule_type -> list of floats (one per board in BOARD_ORDER)
    """
    # Accumulate per (rule, board) across models
    # sums[rule][board_idx] = list of values
    sums = {r: {i: [] for i in range(len(BOARD_ORDER))} for r in RULE_TYPES}

    for mf in MODEL_FOLDERS:
        model_dir = od_dir / "all" / mf
        for bi, bt in enumerate(BOARD_ORDER):
            abt = _load_accuracy_by_type(model_dir, bt)
            if abt is None:
                continue
            for rule in RULE_TYPES:
                if rule in abt:
                    sums[rule][bi].append(abt[rule])

    result = {}
    for rule in RULE_TYPES:
        result[rule] = [
            np.mean(sums[rule][bi]) * 100 if sums[rule][bi] else np.nan
            for bi in range(len(BOARD_ORDER))
        ]
    return result


# ── Chart ─────────────────────────────────────────────────────────────────

def create_rule_detection_chart(od_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(od_dir)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.5))

    x = np.arange(len(BOARD_ORDER))

    for rule in RULE_TYPES:
        vals = data[rule]
        ax.plot(x, vals,
                color=RULE_COLORS[rule],
                marker=RULE_MARKERS[rule],
                markersize=5,
                linewidth=1.3,
                label=RULE_LABELS[rule])

    ax.set_xticks(x)
    ax.set_xticklabels([BOARD_LABELS[bt] for bt in BOARD_ORDER], fontsize=7)
    ax.set_ylabel("Detection Accuracy (\\%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.3, len(BOARD_ORDER) - 0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    ax.legend(
        fontsize=6,
        ncol=4,
        loc="lower right",
        frameon=True,
        edgecolor="0.85",
        fancybox=False,
        columnspacing=0.8,
        handlelength=2.0,
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

    create_rule_detection_chart(od_dir, output_dir / "rule_detection.pdf")


if __name__ == "__main__":
    main()
