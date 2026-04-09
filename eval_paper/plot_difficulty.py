#!/usr/bin/env python3
"""
Line chart showing model accuracy by puzzle difficulty level (1–5)
from the SPaRC evaluation.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_config import setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color

# ── Constants ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "google_gemma-3-27b-it":                          ("gemma-3-27b-it",          "Gemma 3 27B"),
    "google_gemma-4-31B-it":                          ("gemma-4-31B-it",          "Gemma 4 31B"),
    "Qwen_Qwen3.5-27B":                              ("Qwen3.5-27B",             "Qwen 3.5 27B"),
    "QuantTrio_Qwen3.5-397B-A17B-AWQ":               ("Qwen3.5-397B-A17B-AWQ",  "Qwen 3.5 397B"),
    "meta-llama_Llama-4-Scout-17B-16E-Instruct":     ("Llama-4-Scout-17B-16E-Instruct", "Llama 4 Scout"),
    "mistralai_Mistral-Small-3.2-24B-Instruct-2506": ("Mistral-Small-3.2-24B-Instruct-2506", "Mistral Small 3.2"),
    "zai-org_GLM-4.6V":                              ("GLM-4.6V",                "GLM 4.6V"),
}

DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

BOARD_TYPE = "text"


# ── Data loading ──────────────────────────────────────────────────────────

def get_sparc_difficulty(stats_file):
    """Return list of accuracy (%) for difficulty 1-5 from a sparc CSV."""
    df = pd.read_csv(stats_file)
    accs = [np.nan] * 5
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip()
        for lvl in DIFFICULTY_LEVELS:
            if metric == f"Difficulty {lvl} Solved":
                accs[lvl - 1] = float(str(row["Percentage"]).replace("%", ""))
    return accs


def get_test_difficulty(model_dir):
    """Return difficulty accuracy (%) for the text board type, levels 1-5."""
    pattern = f"{BOARD_TYPE}-B_prompt_engineering-P_*_stats_overall.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        return [np.nan] * 5
    with open(matches[-1]) as f:
        data = json.load(f)
    if "error" in data or "avg_accuracy_by_difficulty_level" not in data:
        return [np.nan] * 5
    abd = data["avg_accuracy_by_difficulty_level"]
    return [abd.get(str(lvl), np.nan) * 100.0 if abd.get(str(lvl)) is not None
            else np.nan for lvl in DIFFICULTY_LEVELS]


def collect_data(sparc_dir, test_dir):
    """Return list of dicts with display_name, sparc_accs, test_accs."""
    results = []
    for sparc_stem, (test_folder, display_name) in MODEL_REGISTRY.items():
        sparc_file = sparc_dir / f"{sparc_stem}_vlm_stats.csv"
        if not sparc_file.exists():
            continue

        sparc_accs = get_sparc_difficulty(sparc_file)

        model_dir = test_dir / "all" / test_folder
        test_accs = get_test_difficulty(model_dir)

        results.append({
            "display_name": display_name,
            "sparc_accs": sparc_accs,
            "test_accs": test_accs,
        })

    results.sort(key=lambda d: np.nanmean(d["sparc_accs"]), reverse=True)
    return results


# ── Chart ─────────────────────────────────────────────────────────────────

def create_difficulty_chart(sparc_dir, test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(sparc_dir, test_dir)
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

        ax.plot(x, d["sparc_accs"], color=color, marker=marker,
                markersize=4, linewidth=1.3, linestyle="--", alpha=0.5)

        ax.plot(x, d["test_accs"], color=color, marker=marker,
                markersize=4, linewidth=1.3, label=d["display_name"])

    ax.set_xticks(DIFFICULTY_LEVELS)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(-2, 102)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([], [], color="0.4", linewidth=1.3, linestyle="--", alpha=0.5,
               label="Original evaluation"),
        Line2D([], [], color="0.4", linewidth=1.3,
               label="Controlled evaluation"),
    ]

    model_handles, model_labels = ax.get_legend_handles_labels()
    all_handles = style_handles + model_handles
    all_labels = ["Original evaluation", "Controlled evaluation"] + model_labels

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
        ncol=3,
        frameon=False,
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

    create_difficulty_chart(sparc_dir, test_dir, output_dir / "difficulty_comparison.pdf")


if __name__ == "__main__":
    main()
