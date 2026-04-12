#!/usr/bin/env python3
"""
Bar chart showing absolute accuracy improvement (best board - original)
at each difficulty level, per model.

Reveals that visual representation improvements benefit easy puzzles
far more than hard ones.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from plot_config import setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color

# -- Constants -------------------------------------------------------------

MODEL_REGISTRY = {
    "gemma-3-27b-it":                          "Gemma 3 27B",
    "gemma-4-31B-it":                          "Gemma 4 31B",
    "Qwen3.5-27B":                             "Qwen 3.5 27B",
    "Qwen3.5-397B-A17B-AWQ":                   "Qwen 3.5 397B",
    "Llama-4-Scout-17B-16E-Instruct":          "Llama 4 Scout",
    "Mistral-Small-3.2-24B-Instruct-2506":     "Mistral Small 3.2",
    "GLM-4.6V":                                "GLM 4.6V",
}

BOARD_TYPES = [
    "original",
    "coordinate_grid",
    "start_end_marked",
    "coordinate_grid_and_start_end_marked",
    "path_cell_annotated",
    "text",
]

DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


# -- Data helpers ----------------------------------------------------------

def _read_difficulty_accs(model_dir, board_type):
    pattern = f"{board_type}-B_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data or "avg_accuracy_by_difficulty_level" not in data:
        return None
    abd = data["avg_accuracy_by_difficulty_level"]
    return {int(k): v * 100.0 for k, v in abd.items()}


def collect_data(test_dir):
    """Return {display_name: {difficulty: improvement_pp}}."""
    result = {}
    for folder, display in MODEL_REGISTRY.items():
        model_dir = test_dir / "all" / folder
        if not model_dir.exists():
            continue

        orig_accs = _read_difficulty_accs(model_dir, "original")
        if orig_accs is None:
            continue

        # Find best board per difficulty
        best_accs = dict(orig_accs)
        for bt in BOARD_TYPES:
            bt_accs = _read_difficulty_accs(model_dir, bt)
            if bt_accs is None:
                continue
            for lvl in DIFFICULTY_LEVELS:
                if lvl in bt_accs and bt_accs[lvl] > best_accs.get(lvl, 0):
                    best_accs[lvl] = bt_accs[lvl]

        improvements = {}
        for lvl in DIFFICULTY_LEVELS:
            o = orig_accs.get(lvl, 0)
            b = best_accs.get(lvl, 0)
            improvements[lvl] = b - o

        result[display] = improvements

    return result


# -- Chart -----------------------------------------------------------------

def create_improvement_by_difficulty_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    models = list(data.keys())
    n_models = len(models)
    n_levels = len(DIFFICULTY_LEVELS)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.85))

    x = np.arange(n_levels)

    for mi, model in enumerate(models):
        color = get_model_color(model)
        marker = MARKERS[mi % len(MARKERS)]
        vals = [data[model].get(lvl, 0) for lvl in DIFFICULTY_LEVELS]
        ax.plot(x, vals, color=color, marker=marker, markersize=5,
                linewidth=1.5, label=model, zorder=3)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="-", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {d}" for d in DIFFICULTY_LEVELS], fontsize=7)
    ax.set_ylabel("Improvement (pp)")
    ax.set_xlabel("Difficulty Level")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(fontsize=6, loc="upper right", frameon=True,
              framealpha=0.95, fancybox=False,
              handletextpad=0.4, handlelength=1.5)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_improvement_by_difficulty_chart(
        test_dir, output_dir / "improvement_by_difficulty.pdf")


if __name__ == "__main__":
    main()
