#!/usr/bin/env python3
"""
Chart showing average completion-token generation per task,
broken down by board type and model, across difficulty levels 1–5.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, get_model_color

# ── Constants ─────────────────────────────────────────────────────────────

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

BOARD_LABELS = {
    "original":                                "Original",
    "coordinate_grid":                         "Coord. Grid",
    "start_end_marked":                        "Start/End",
    "coordinate_grid_and_start_end_marked":    "Coord. + S/E",
    "path_cell_annotated":                     "Cell Annot.",
    "text":                                    "Text",
}

BOARD_COLORS = {
    "original":                                "#E53935",
    "coordinate_grid":                         "#1E88E5",
    "start_end_marked":                        "#43A047",
    "coordinate_grid_and_start_end_marked":    "#FB8C00",
    "path_cell_annotated":                     "#8E24AA",
    "text":                                    "#00897B",
}

BOARD_MARKERS = {
    "original":                                "o",
    "coordinate_grid":                         "s",
    "start_end_marked":                        "D",
    "coordinate_grid_and_start_end_marked":    "^",
    "path_cell_annotated":                     "v",
    "text":                                    "P",
}

DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]


# ── Data loading ──────────────────────────────────────────────────────────

def _read_tokens_by_difficulty(model_dir, board_type):
    """Return dict {difficulty_level: avg_completion_tokens} or None."""
    pattern = f"{board_type}-B_prompt_engineering-P_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data:
        return None
    abd = data.get("avg_tokens_per_task_by_difficulty_level")
    if not abd:
        return None
    return {int(k): v["completion_tokens"] for k, v in abd.items()}


def collect_data(test_dir):
    """Return nested dict: model_name -> board_type -> [tokens per difficulty 1-5]."""
    results = {}
    for folder, display_name in MODEL_REGISTRY.items():
        model_dir = test_dir / "all" / folder
        if not model_dir.exists():
            continue
        model_data = {}
        for bt in BOARD_TYPES:
            tokens = _read_tokens_by_difficulty(model_dir, bt)
            if tokens is None:
                continue
            model_data[bt] = [tokens.get(lvl, np.nan) for lvl in DIFFICULTY_LEVELS]
        if model_data:
            results[display_name] = model_data
    return results


# ── Chart ─────────────────────────────────────────────────────────────────

def create_token_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    models = sorted(data.keys(),
                    key=lambda m: np.nanmean(
                        [np.nanmean(v) for v in data[m].values()]),
                    reverse=True)

    n_models = len(models)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.28 * n_rows),
        sharey=True, sharex=True,
    )
    axes = np.atleast_2d(axes)

    x = np.array(DIFFICULTY_LEVELS)

    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        for bt in BOARD_TYPES:
            if bt not in data[model]:
                continue
            vals = data[model][bt]
            ax.plot(x, vals,
                    color=BOARD_COLORS[bt],
                    marker=BOARD_MARKERS[bt],
                    markersize=3, linewidth=1.0,
                    label=BOARD_LABELS[bt])

        ax.set_title(model, fontsize=7, pad=3)
        ax.set_xticks(DIFFICULTY_LEVELS)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    for row_idx in range(n_rows):
        axes[row_idx][0].set_ylabel("Tokens", fontsize=7)
    for col_idx in range(n_cols):
        last_row = n_rows - 1
        if not axes[last_row][col_idx].get_visible():
            last_row -= 1
        axes[last_row][col_idx].set_xlabel("Difficulty", fontsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=7,
        ncol=len(BOARD_TYPES),
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.1)

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
