#!/usr/bin/env python3
"""
Grouped bar chart comparing prompt types (default, default + text repr,
prompt engineering) across board types.  Values are averaged over all
models that have results for a given prompt × board combination.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES

# ── Constants ─────────────────────────────────────────────────────────────

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
    "coordinate_grid":                       "Coord.\nGrid",
    "start_end_marked":                      "Start/End\nMarked",
    "coordinate_grid_and_start_end_marked":  "Coord. Grid\n+ S/E",
    "path_cell_annotated":                   "Cell\nAnnotated",
    "text":                                  "Text",
}

PROMPT_TYPES = ["default_no_tr", "default_tr", "prompt_engineering"]

PROMPT_LABELS = {
    "default_no_tr":      "Default",
    "default_tr":         "Default + Text Repr.",
    "prompt_engineering":  "Prompt Engineering",
}

PROMPT_COLORS = {
    "default_no_tr":      "#8DA0CB",
    "default_tr":         "#66C2A5",
    "prompt_engineering":  "#FC8D62",
}

PROMPT_HATCHES = {
    "default_no_tr":      "",
    "default_tr":         "",
    "prompt_engineering":  "",
}


# ── Data loading ──────────────────────────────────────────────────────────

def _read_accuracy(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if "error" in data or "accuracy" not in data:
        return None
    return data["accuracy"] * 100.0


MODEL_FOLDER = "Qwen3.5-397B-A17B-AWQ"
SPARC_CSV_STEM = "QuantTrio_Qwen3.5-397B-A17B-AWQ"


def get_sparc_accuracy(sparc_dir):
    """Return SPaRC baseline accuracy (%) for the target model."""
    stats_file = sparc_dir / f"{SPARC_CSV_STEM}_vlm_stats.csv"
    if not stats_file.exists():
        return None
    df = pd.read_csv(stats_file)
    for _, row in df.iterrows():
        if row["Metric"] == "Correctly Solved":
            return float(str(row["Percentage"]).replace("%", ""))
    return None


def collect_data(test_dir):
    """For each prompt_type × board_type, read accuracy for the target model.

    Returns:
        dict  prompt_type -> list[float|nan] (one per board in BOARD_ORDER)
    """
    model_dir = test_dir / "all" / MODEL_FOLDER
    if not model_dir.exists():
        return {}

    result = {}
    for pt in PROMPT_TYPES:
        vals = []
        for bt in BOARD_ORDER:
            pattern = f"{bt}-B_{pt}-P_*_stats_overall.json"
            matches = sorted(model_dir.glob(pattern))
            if matches:
                val = _read_accuracy(matches[-1])
                vals.append(val if val is not None else np.nan)
            else:
                vals.append(np.nan)
        result[pt] = vals
    return result


# ── Chart ─────────────────────────────────────────────────────────────────

def create_prompt_comparison_chart(test_dir, sparc_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)
    if not data:
        print("No data found!")
        return None

    sparc_acc = get_sparc_accuracy(sparc_dir)

    n_boards = len(BOARD_ORDER)
    n_prompts = len(PROMPT_TYPES)
    bar_width = 0.25
    x = np.arange(n_boards)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.4))

    for i, pt in enumerate(PROMPT_TYPES):
        vals = data[pt]
        offsets = x + (i - (n_prompts - 1) / 2) * bar_width
        bars = ax.bar(
            offsets,
            [v if not np.isnan(v) else 0 for v in vals],
            width=bar_width,
            color=PROMPT_COLORS[pt],
            hatch=PROMPT_HATCHES[pt],
            label=PROMPT_LABELS[pt],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold")

    max_val = max(v for vals in data.values() for v in vals if not np.isnan(v))
    y_top = max(max_val, sparc_acc or 0) * 1.18

    if sparc_acc is not None:
        ax.axhline(sparc_acc, color="#333333", linewidth=1.2, linestyle="--",
                    zorder=3, label=f"SPaRC Baseline ({sparc_acc:.1f}\\%)")

    ax.set_xticks(x)
    ax.set_xticklabels([BOARD_LABELS[bt] for bt in BOARD_ORDER], fontsize=7)
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(0, y_top)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    ax.legend(
        fontsize=7,
        ncol=n_prompts + 1,
        loc="upper left",
        frameon=True,
        edgecolor="0.85",
        fancybox=False,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")

    return fig


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"
    sparc_dir = base / "sparc"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_prompt_comparison_chart(test_dir, sparc_dir, output_dir / "prompt_comparison.pdf")


if __name__ == "__main__":
    main()
