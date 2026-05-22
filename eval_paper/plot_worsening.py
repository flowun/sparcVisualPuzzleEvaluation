#!/usr/bin/env python3
"""
Single-column difference chart for the worsening study.

Vertical bars show accuracy delta in percentage points, with each
worsening condition paired against the path-cell-annotated recovery
variant immediately to its right:

  - Worsening row (e.g. Low Contrast): Δ vs the Original board.
  - Recovery row (e.g. + Cell Coord.): Δ vs the Cell Coord. board.

When the recovery bar is much smaller in magnitude than the worsening
bar above it, the cell-coordinate annotation has rescued most of the
accuracy the worsening would otherwise have cost.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import AnnotationBbox
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_color, DEFAULT_PROMPT,
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
        # Color swatch on the left half of the handle box.
        swatch_w = width * 0.40
        swatch = mpatches.Rectangle(
            (xdescent, ydescent), swatch_w, height,
            facecolor=self._color, edgecolor="white", linewidth=0.4,
            transform=trans,
        )
        artists.append(swatch)
        # Logo on the right half (uses the project's properly-zoomed imagebox).
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

# (board_to_show, x-tick label, baseline_board_for_delta).
CONDITIONS = [
    ("low_contrast",                            "Low\nContrast",   "original"),
    ("low_contrast_and_path_cell_annotated",    "+ Cell\nCoord.",  "path_cell_annotated"),
    ("low_resolution",                          "Low\nResolution", "original"),
    ("low_resolution_and_path_cell_annotated",  "+ Cell\nCoord.",  "path_cell_annotated"),
    ("rotated",                                 "Rotated",          "original"),
    ("rotated_and_path_cell_annotated",         "+ Cell\nCoord.",  "path_cell_annotated"),
]

# Dashed dividers between consecutive worsening groups.
GROUP_DIVIDERS = [1.5, 3.5]


def collect_data(test_dir):
    boards = {"original", "path_cell_annotated"}
    for board, _, baseline in CONDITIONS:
        boards.add(board)
        boards.add(baseline)

    result = {}
    for folder, _ in MODELS:
        model_dir = test_dir / "all" / folder
        vals = {}
        for bt in boards:
            vals[bt] = read_json_metric(model_dir, bt, "accuracy",
                                        prompt_type=DEFAULT_PROMPT)
        result[folder] = vals
    return result


def create_worsening_chart(test_dir, output_path=None):
    setup_plot_style(use_latex=True)

    data = collect_data(test_dir)

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

    all_deltas = []
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
        deltas = []
        for board, _, baseline in CONDITIONS:
            v = vals[board]
            b = vals[baseline]
            d = v - b if not np.isnan(v) and not np.isnan(b) else 0.0
            deltas.append(d)
            all_deltas.append(d)
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        ax.bar(
            offsets,
            deltas,
            width=bar_width,
            color=color,
            label=legend_labels[-1],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        for off, d in zip(offsets, deltas):
            if abs(d) < 0.05:
                continue
            sign = "+" if d > 0 else ""
            txt_offset = 0.2 if d > 0 else -0.2
            va = "bottom" if d > 0 else "top"
            ax.text(off, d + txt_offset, f"{sign}{d:.1f}",
                    ha="center", va=va,
                    fontsize=5.5, fontweight="bold", color=color)

    y_lim = max(abs(min(all_deltas)), abs(max(all_deltas))) * 1.22
    ax.set_ylim(-y_lim, y_lim)

    # Zero reference line.
    ax.axhline(0, color="#333333", linewidth=0.9, zorder=1.5)

    # Worsening-type label centered above each (worsening, recovery) pair.
    worsening_groups = [
        ((0, 1), "Low Contrast"),
        ((2, 3), "Low Resolution"),
        ((4, 5), "Rotated"),
    ]
    for (start, end), label in worsening_groups:
        ax.text((start + end) / 2, y_lim * 0.96, label,
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
    ax.set_ylabel(r"$\Delta$ Accuracy (\%)", fontsize=8, labelpad=2)

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

    ax.legend(
        legend_handles,
        legend_labels,
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
    test_dir = base / "test"

    output_dir = base / "figures"
    output_dir.mkdir(exist_ok=True)

    create_worsening_chart(test_dir, output_dir / "worsening_comparison.pdf")


if __name__ == "__main__":
    main()
