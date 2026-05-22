#!/usr/bin/env python3
"""
Single-column difference chart for the object-detection worsening study.

Mirrors plot_worsening.py: paired bars per worsening type show the bare
worsening (Δ vs Original) next to the path-cell-annotated recovery
(Δ vs Cell Coord.). Smaller-magnitude recovery bars indicate cell
coordinates rescue most of the rule-detection accuracy lost to that
worsening. Uses Qwen 3.5 397B and Gemma 4 31B.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color,
    read_json_metric, get_model_imagebox,
)


class _HandlerColorLogo(HandlerBase):
    """Render a color swatch followed by the model logo as the legend handle."""

    def __init__(self, display_name, color, zoom_factor=0.4):
        super().__init__()
        self._display_name = display_name
        self._color = color
        self._zoom_factor = zoom_factor

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        swatch_w = width * 0.40
        swatch = mpatches.Rectangle(
            (xdescent, ydescent), swatch_w, height,
            facecolor=self._color, edgecolor="white", linewidth=0.4,
            transform=trans,
        )
        artists.append(swatch)
        oi = get_model_imagebox(self._display_name, zoom_factor=self._zoom_factor)
        if oi is not None:
            ab = AnnotationBbox(
                oi,
                (xdescent + width * 0.78, ydescent + height / 2),
                xycoords=trans, frameon=False, pad=0,
            )
            artists.append(ab)
        return artists


MODELS = [
    ("Qwen3.5-397B-A17B-AWQ", "Qwen 3.5 397B"),
    ("gemma-4-31B-it",         "Gemma 4 31B"),
]

# (board, x-tick label baseline, baseline_board_for_delta).
CONDITIONS = [
    ("low_contrast",                            "Low\nContrast",   "original"),
    ("low_contrast_and_path_cell_annotated",    "+ Cell\nCoord.",  "path_cell_annotated"),
    ("low_resolution",                          "Low\nResolution", "original"),
    ("low_resolution_and_path_cell_annotated",  "+ Cell\nCoord.",  "path_cell_annotated"),
    ("rotated",                                 "Rotated",          "original"),
    ("rotated_and_path_cell_annotated",         "+ Cell\nCoord.",  "path_cell_annotated"),
]

GROUP_DIVIDERS = [1.5, 3.5]


def collect_data(od_dir):
    boards = {"original", "path_cell_annotated"}
    for board, _, baseline in CONDITIONS:
        boards.add(board)
        boards.add(baseline)

    result = {}
    for folder, _ in MODELS:
        model_dir = od_dir / "all" / folder
        vals = {}
        for bt in boards:
            vals[bt] = read_json_metric(model_dir, bt, "fraction_average",
                                        prompt_type="default")
        result[folder] = vals
    return result


def create_od_worsening_chart(od_dir, output_path=None, mode="delta"):
    """Render the OD worsening study chart.

    mode:
        "delta"    — bars show v − baseline in percentage points.
        "absolute" — bars show absolute rule-detection accuracy with a short
                     black dashed segment per bar marking the matching
                     baseline. Bar labels show the signed delta.
    """
    setup_plot_style(use_latex=True)

    data = collect_data(od_dir)

    n_conds = len(CONDITIONS)
    n_models = len(MODELS)
    bar_width = 0.36

    x = np.arange(n_conds)

    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.55)
    )

    # Subtle shading on the "+ Cell Coord." columns to distinguish them
    # from the bare worsening columns next to them.
    for col_x in [1, 3, 5]:
        ax.axvspan(col_x - 0.5, col_x + 0.5,
                   color="#EBEEF2", alpha=0.7, zorder=0)

    all_vals = []
    legend_labels = []
    for i, (folder, display_name) in enumerate(MODELS):
        vals = data[folder]
        color = get_model_color(display_name)
        orig = vals["original"]
        cc = vals["path_cell_annotated"]
        legend_labels.append(
            f"{display_name} "
            f"(Orig.: {orig:.1f}\\%, Cell Coord.: {cc:.1f}\\%)"
        )
        bar_vals = []
        baseline_vals = []
        for board, _, baseline in CONDITIONS:
            v = vals[board]
            b = vals[baseline]
            baseline_vals.append(b)
            if np.isnan(v) or np.isnan(b):
                bar_vals.append(0.0)
                continue
            if mode == "delta":
                bar_vals.append(v - b)
            elif mode == "absolute":
                bar_vals.append(v)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        all_vals.extend(bar_vals)
        if mode == "absolute":
            all_vals.extend([b for b in baseline_vals if not np.isnan(b)])
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        ax.bar(
            offsets, bar_vals, width=bar_width,
            color=color, label=legend_labels[-1],
            edgecolor="white", linewidth=0.4, zorder=2,
        )

        if mode == "absolute":
            for off, b in zip(offsets, baseline_vals):
                if np.isnan(b):
                    continue
                ax.hlines(
                    b,
                    off - bar_width / 2,
                    off + bar_width / 2,
                    colors="black", linestyles=(0, (2.5, 1.5)),
                    linewidth=1.2, zorder=5,
                )

        for off, v_bar, b_val in zip(offsets, bar_vals, baseline_vals):
            if mode == "absolute":
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
                continue
            if abs(v_bar) < 0.05:
                continue
            sign = "+" if v_bar > 0 else ""
            step = max(abs(min(all_vals)), abs(max(all_vals))) * 0.02
            txt_offset = step if v_bar > 0 else -step
            va = "bottom" if v_bar > 0 else "top"
            ax.text(off, v_bar + txt_offset, f"{sign}{v_bar:.1f}",
                    ha="center", va=va,
                    fontsize=5.5, fontweight="bold", color=color)

    if mode == "absolute":
        y_lim_top = max(all_vals) * 1.32
        ax.set_ylim(0, y_lim_top)
        ax.set_yticks(np.arange(0, 101, 20))
        text_y = y_lim_top * 0.98
    else:
        y_lim = max(abs(min(all_vals)), abs(max(all_vals))) * 1.22
        ax.set_ylim(-y_lim, y_lim)
        ax.axhline(0, color="#333333", linewidth=0.9, zorder=1.5)
        text_y = y_lim * 0.96

    # Worsening-type label centered above each (worsening, recovery) pair.
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
    if mode == "absolute":
        ax.set_ylabel(r"Rule Detection (\%)", fontsize=8, labelpad=2)
    else:
        ax.set_ylabel(r"$\Delta$ Rule Detection (\%)", fontsize=8, labelpad=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    # Per-model logo zoom (Gemma is rendered slightly larger).
    LOGO_ZOOM = {
        "Qwen 3.5 397B": 0.55,
        "Gemma 4 31B":   0.75,
    }

    legend_handles = []
    handler_map = {}
    for (folder, display_name), label in zip(MODELS, legend_labels):
        color = get_model_color(display_name)
        h = mpatches.Patch(color=color, label=label)
        legend_handles.append(h)
        handler_map[h] = _HandlerColorLogo(
            display_name, color,
            zoom_factor=LOGO_ZOOM.get(display_name, 0.55),
        )

    extra_handles = []
    extra_labels = []
    if mode == "absolute":
        extra_handles.append(
            Line2D([0], [0], color="black",
                   linestyle=(0, (2.5, 1.5)), linewidth=1.2)
        )
        extra_labels.append("Matching baseline accuracy")

    ax.legend(
        legend_handles + extra_handles,
        legend_labels + extra_labels,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=1,
        fontsize=6.5,
        frameon=False,
        handletextpad=0.6,
        handlelength=2.6,
        labelspacing=0.30,
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

    create_od_worsening_chart(od_dir, output_dir / "od_worsening_comparison.pdf",
                              mode="delta")
    create_od_worsening_chart(od_dir, output_dir / "od_worsening_comparison_absolute.pdf",
                              mode="absolute")


if __name__ == "__main__":
    main()
