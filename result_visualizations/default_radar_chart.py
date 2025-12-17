import os
import numpy as np
import matplotlib.pyplot as plt

def create_radar_chart(
    data,
    dimensions=None,
    title="",
    figsize=(8, 8),
    colors=None,
    fill=True,
    fill_alpha=0.05,
    legend=True,
    grid=True,
    output_path='',
    save_format='pdf',
    transparent=False,
    rmax=1.0,
    rticks=(0.25, 0.50, 0.75, 1.00),
    rlabel_format="percent",   # "percent" or "float"
    start_angle_deg=90 - 18,
    clockwise=False,
    line_width=2.2,
    marker_size=7,
    marker_styles=("o", "s", "P", "^", "D", "X"),  # circle, square, plus, triangle...
    dimension_fontsize=16,
    dimension_fontweight="bold",
    tick_fontsize=11,
    title_fontsize=22,
    legend_fontsize=16,
    show_single_legend=False,
):

    if dimensions is None:
        dimensions = ["Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5"]

    num_dimensions = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_dimensions, endpoint=False).tolist()
    angles += angles[:1]

    labels = list(data.keys())
    values_list = []

    for label in labels:
        values = data[label]
        if isinstance(values, dict):
            values = [values.get(dim, 0) for dim in dimensions]
        if len(values) != num_dimensions:
            raise ValueError(f"Dataset '{label}' does not match {num_dimensions} dimensions")
        values_list.append(values + values[:1])

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

    # --- orientation ---
    ax.set_theta_offset(np.deg2rad(start_angle_deg))
    ax.set_theta_direction(-1 if clockwise else 1)

    # --- category labels (big + bold) ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=dimension_fontsize, fontweight=dimension_fontweight)

    # --- radial scale + percent tick labels (25/50/75/100%) ---
    ax.set_ylim(0, rmax)
    ax.set_yticks(list(rticks))
    if rlabel_format == "percent":
        ax.set_yticklabels([f"{int(round(t * 100))}%" for t in rticks], fontsize=tick_fontsize, color="0.35")
    else:
        ax.set_yticklabels([f"{t:.2f}" for t in rticks], fontsize=tick_fontsize, color="0.35")

    # place radial tick labels near the right side
    ax.set_rlabel_position(22.5)

    # --- grid / frame styling to match the soft gray rings ---
    if not grid:
        ax.grid(False)
    else:
        ax.grid(color="0.82", linestyle="-", linewidth=0.9)
    ax.spines["polar"].set_color("0.82")
    ax.spines["polar"].set_linewidth(1.0)

    # --- series plotting: markers + translucent fills ---
    color_cycle = colors or colors or plt.cm.tab10.colors[1:]
    for idx, (label, values) in enumerate(zip(labels, values_list)):
        line_color = color_cycle[idx % len(color_cycle)]
        marker = marker_styles[idx % len(marker_styles)]

        ax.plot(
            angles,
            values,
            linewidth=line_width,
            color=line_color,
            label=label,
            marker=marker,
            markersize=marker_size,
            markerfacecolor=line_color,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        if fill:
            ax.fill(angles, values, color=line_color, alpha=fill_alpha)

    # title: keep, but make it less dominant (image focuses on labels/legend)
    ax.set_title(title, fontsize=title_fontsize, pad=18)

    # --- legend at the bottom ---
    if legend and (len(labels) > 1 or show_single_legend):
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.2,
            handletextpad=0.5,
            columnspacing=1.8,
        )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0,
            format=save_format,
        )

    plt.show()

# Example usage
if __name__ == "__main__":
    sample_data = {
        "Setup A": [0.9, 0.8, 0.85, 0.7, 0.95],
        "Setup B": [0.75, 0.9, 0.8, 0.85, 0.65],
        "Setup C": [0.6, 0.7, 0.75, 0.8, 0.7],
    }
    dimensions = ["Incorrect\nStart/End", "Invalid\nPath", "Rule Cell\nCrossing", "Intersecting\nLine", "Disconnected\nLine"]

    create_radar_chart(
        data=sample_data,
        dimensions=dimensions,
        # title="Setup Performance Comparison",
        # output_path='images/radar_chart_example.pdf',
        transparent=False
    )