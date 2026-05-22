#!/usr/bin/env python3
"""
GRPO training effect on Qwen 3 VL Thinking (4B and 8B) for SPaRC accuracy,
combined into a single chart.

Three image types on the x-axis (Original, Text Annotation, No Image),
each with four bars: 4B Untrained / 4B GRPO / 8B Untrained / 8B GRPO.
Untrained bars use diagonal hatching, GRPO bars are solid; lighter and
darker purple distinguish the two model sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import AnnotationBbox
from pathlib import Path
from plot_config import (
    setup_plot_style, COLUMN_WIDTH_INCHES, get_model_imagebox,
)


# (Untrained, GRPO) accuracy in percent.
DATA = {
    "Qwen 3 VL 4B Thinking": {
        "Baseline":     (7.8, 10.8),
        "Original":     (1.4, 0.8),
        "Text Symbols": (1.4, 5.2),
    },
    "Qwen 3 VL 8B Thinking": {
        "Baseline":     (10.0, 12.8),
        "Original":     (0.2, 0.4),
        "Text Symbols": (1.2, 5.8),
    },
}

IMAGE_TYPES = ["Baseline", "Original", "Text Symbols"]
MODELS = ["Qwen 3 VL 4B Thinking", "Qwen 3 VL 8B Thinking"]

MODEL_COLORS = {
    "Qwen 3 VL 4B Thinking": "#B39DDB",  # lighter purple
    "Qwen 3 VL 8B Thinking": "#5E35B1",  # darker purple
}

LOGO_ZOOM = {
    "Qwen 3 VL 4B Thinking": 0.55,
    "Qwen 3 VL 8B Thinking": 0.55,
}


class _HandlerColorLogo(HandlerBase):
    """Color swatch + model logo as the legend handle."""

    def __init__(self, display_name, color, hatch="", zoom_factor=0.55):
        super().__init__()
        self._display_name = display_name
        self._color = color
        self._hatch = hatch
        self._zoom_factor = zoom_factor

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        artists = []
        swatch_w = width * 0.40
        swatch = mpatches.Rectangle(
            (xdescent, ydescent), swatch_w, height,
            facecolor=self._color, edgecolor="white", linewidth=0.4,
            hatch=self._hatch,
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


def main():
    setup_plot_style(use_latex=True)

    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.45)
    )

    n_groups = len(IMAGE_TYPES)
    x = np.arange(n_groups)
    bar_width = 0.18

    # Within each image-type group: 4B-U, 4B-G, 8B-U, 8B-G.
    bar_specs = [
        ("Qwen 3 VL 4B Thinking", 0, -0.30, "///"),  # Untrained
        ("Qwen 3 VL 4B Thinking", 1, -0.10, ""),     # GRPO
        ("Qwen 3 VL 8B Thinking", 0, +0.10, "///"),  # Untrained
        ("Qwen 3 VL 8B Thinking", 1, +0.30, ""),     # GRPO
    ]

    all_vals = []
    for model, cond_idx, offset, hatch in bar_specs:
        color = MODEL_COLORS[model]
        vals = [DATA[model][img][cond_idx] for img in IMAGE_TYPES]
        all_vals.extend(vals)
        bars = ax.bar(
            x + offset, vals, width=bar_width,
            color=color, hatch=hatch,
            edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=6, fontweight="bold",
                color=color,
            )

    # Group dividers between image types.
    for sep_x in [0.5, 1.5]:
        ax.axvline(sep_x, color="#999999", linewidth=0.5, linestyle="--",
                   alpha=0.55, zorder=1)

    y_max = max(all_vals) * 1.15
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(IMAGE_TYPES, fontsize=8)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(r"Accuracy (\%)", fontsize=9, labelpad=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    # Legend: 2 columns. Left col = condition (hatched / solid). Right col =
    # model (color swatch + logo). Column-major fill puts Untrained and GRPO
    # in the left column, 4B and 8B in the right.
    GRAY = "#888888"
    h_untrained = mpatches.Patch(facecolor=GRAY, hatch="///",
                                 edgecolor="white", linewidth=0.4)
    h_grpo = mpatches.Patch(facecolor=GRAY, edgecolor="white", linewidth=0.4)
    h_4b = mpatches.Patch(color=MODEL_COLORS["Qwen 3 VL 4B Thinking"])
    h_8b = mpatches.Patch(color=MODEL_COLORS["Qwen 3 VL 8B Thinking"])

    handles = [h_untrained, h_grpo, h_4b, h_8b]
    labels = [
        "Untrained", "GRPO",
        "Qwen 3 VL 4B Thinking", "Qwen 3 VL 8B Thinking",
    ]

    handler_map = {
        h_4b: _HandlerColorLogo(
            "Qwen 3 VL 4B Thinking",
            MODEL_COLORS["Qwen 3 VL 4B Thinking"],
            zoom_factor=LOGO_ZOOM["Qwen 3 VL 4B Thinking"],
        ),
        h_8b: _HandlerColorLogo(
            "Qwen 3 VL 8B Thinking",
            MODEL_COLORS["Qwen 3 VL 8B Thinking"],
            zoom_factor=LOGO_ZOOM["Qwen 3 VL 8B Thinking"],
        ),
    }

    ax.legend(
        handles, labels,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=7.5,
        frameon=False,
        handletextpad=0.6,
        handlelength=2.4,
        labelspacing=0.30,
        columnspacing=1.4,
    )

    output_dir = Path(__file__).parent.parent / "evaluation" / "results" / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "grpo_text_only.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    main()
