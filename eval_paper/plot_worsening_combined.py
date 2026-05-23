#!/usr/bin/env python3
"""Two-panel worsening comparison combining the task-accuracy and complete
object-detection absolute charts side by side.

Each panel shows bars at the absolute accuracy with a short black dashed
segment marking the matching baseline; bar labels show the signed delta
from that baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

from plot_config import (
    setup_plot_style, TEXT_WIDTH_INCHES, COLUMN_WIDTH_INCHES, get_model_color,
)
from plot_worsening import (
    MODELS, CONDITIONS, GROUP_DIVIDERS,
    collect_data as collect_task_data,
    _HandlerColorLogo,
)
from plot_od_worsening import collect_data as collect_od_data


LOGO_ZOOM = {
    "Qwen 3.5 397B": 0.55,
    "Gemma 4 31B":   0.75,
}


def _draw_panel(ax, data, title, ylabel):
    n_conds = len(CONDITIONS)
    n_models = len(MODELS)
    bar_width = 0.36
    x = np.arange(n_conds)

    for col_x in [1, 3, 5]:
        ax.axvspan(col_x - 0.5, col_x + 0.5,
                   color="#EBEEF2", alpha=0.7, zorder=0)

    all_vals = []
    for i, (folder, display_name) in enumerate(MODELS):
        vals = data[folder]
        color = get_model_color(display_name)
        bar_vals = []
        baseline_vals = []
        for board, _, baseline in CONDITIONS:
            v = vals[board]
            b = vals[baseline]
            baseline_vals.append(b)
            if np.isnan(v) or np.isnan(b):
                bar_vals.append(0.0)
            else:
                bar_vals.append(v)
        all_vals.extend(bar_vals)
        all_vals.extend([b for b in baseline_vals if not np.isnan(b)])
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        ax.bar(offsets, bar_vals, width=bar_width,
               color=color, edgecolor="white", linewidth=0.4, zorder=2)

        for off, b in zip(offsets, baseline_vals):
            if np.isnan(b):
                continue
            ax.hlines(
                b,
                off - bar_width / 2 - 0.02,
                off + bar_width / 2 + 0.02,
                colors="black", linestyles=(0, (2.5, 1.5)),
                linewidth=1.2, zorder=5,
            )

        for off, v_bar, b_val in zip(offsets, bar_vals, baseline_vals):
            if v_bar < 0.05:
                continue
            if np.isnan(b_val):
                txt = f"{v_bar:.1f}"
            else:
                d = v_bar - b_val
                sign = "+" if d > 0 else ""
                txt = f"{sign}{d:.1f}"
            txt_offset = max(all_vals) * 0.018
            anchor_y = v_bar
            if not np.isnan(b_val) and b_val > v_bar:
                anchor_y = b_val
            ax.text(off, anchor_y + txt_offset, txt,
                    ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold", color=color)

    y_lim_top = max(all_vals) * 1.32
    ax.set_ylim(0, y_lim_top)
    ax.set_yticks(np.arange(0, 101, 20))
    text_y = y_lim_top * 0.98

    worsening_groups = [
        ((0, 1), "Low Contrast"),
        ((2, 3), "Low Resolution"),
        ((4, 5), "Rotated"),
    ]
    for (start, end), label in worsening_groups:
        ax.text((start + end) / 2, text_y, label,
                ha="center", va="top",
                fontsize=7, color="#444444", fontstyle="italic",
                fontweight="bold")

    for sep_x in GROUP_DIVIDERS:
        ax.axvline(sep_x, color="#999999", linewidth=0.5, linestyle="--",
                   alpha=0.55, zorder=1)

    ax.set_xticks(x)
    x_tick_labels = [
        "Original" if baseline == "original" else "Cell\nCoord."
        for _, _, baseline in CONDITIONS
    ]
    ax.set_xticklabels(x_tick_labels, fontsize=6.5)
    ax.set_xlim(-0.5, n_conds - 0.5)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    ax.set_title(title, fontsize=8, pad=2, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)


def main():
    base = Path(__file__).parent.parent / "evaluation" / "results"
    test_dir = base / "test"
    od_dir = base / "object_detection" / "test"

    setup_plot_style(use_latex=True)
    task_data = collect_task_data(test_dir)
    od_data = collect_od_data(od_dir, metric="accuracy")

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(TEXT_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.55),
        gridspec_kw={"wspace": 0.18},
    )

    _draw_panel(ax_left, task_data, "Task Accuracy", r"Accuracy (\%)")
    _draw_panel(ax_right, od_data, "Complete Board Detection",
                r"Board Detection (\%)")

    legend_handles = []
    legend_labels = []
    handler_map = {}
    for folder, display_name in MODELS:
        color = get_model_color(display_name)
        t_orig = task_data[folder]["original"]
        t_cc = task_data[folder]["path_cell_annotated"]
        o_orig = od_data[folder]["original"]
        o_cc = od_data[folder]["path_cell_annotated"]
        h = mpatches.Patch(color=color)
        label = (
            f"{display_name} — "
            f"Task Orig./Cell Coord.: {t_orig:.1f}/{t_cc:.1f}\\%, "
            f"OD: {o_orig:.1f}/{o_cc:.1f}\\%"
        )
        legend_handles.append(h)
        legend_labels.append(label)
        handler_map[h] = _HandlerColorLogo(
            display_name, color,
            zoom_factor=LOGO_ZOOM.get(display_name, 0.55),
        )

    legend_handles.append(
        Line2D([0], [0], color="black",
               linestyle=(0, (2.5, 1.5)), linewidth=1.2)
    )
    legend_labels.append("Matching baseline accuracy")

    fig.legend(
        legend_handles, legend_labels,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        fontsize=6.5,
        frameon=False,
        handletextpad=0.6,
        handlelength=2.6,
        labelspacing=0.30,
    )

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "worsening_comparison_combined.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    main()
