import json
import os
import matplotlib.pyplot as plt
from util import BAR_LABEL_COLORS, get_bar_color


def parse_eval_key(filename):
    base = filename.replace("_stats_individual.json", "")
    board_part, remainder = base.split("-B_", 1)
    prompt_part = remainder.split("-P_", 1)[0]
    return board_part, prompt_part


def load_individual_results(evaluation_dir, selection_filter=None):
    runs = {}
    for file in sorted(os.listdir(evaluation_dir)):
        if not file.endswith("_stats_individual.json"):
            continue
        board_type, prompt_type = parse_eval_key(file)
        if selection_filter is not None and (board_type, prompt_type) not in selection_filter:
            continue
        with open(os.path.join(evaluation_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        puzzles = list(data.values()) if isinstance(data, dict) else data
        runs.setdefault((board_type, prompt_type), []).extend(puzzles)
    return runs


def compute_accuracy_by_board_size(individual_results):
    accuracy_by_size = {}
    for key, puzzles in individual_results.items():
        size_counts = {}
        for puzzle in puzzles:
            width = puzzle.get("width")
            height = puzzle.get("height")
            if width is None or height is None:
                continue
            area = width * height
            counts = size_counts.setdefault(area, {"correct": 0, "total": 0})
            if puzzle.get("is_valid"):
                counts["correct"] += 1
            counts["total"] += 1
        accuracy_by_size[key] = {
            area: counts["correct"] / counts["total"] if counts["total"] else 0
            for area, counts in size_counts.items()
        }
    return accuracy_by_size


def create_board_size_line_chart(
    accuracy_by_size,
    title="Accuracy by Board Size",
    x_label="Board Size (width x height)",
    y_label="Accuracy (%)",
    output_path="",
    figsize=(8, 5),
    save_format="pdf",
    transparent=False,
    y_limit=None,
    title_fontsize=12,
    label_colors=None,
):
    if not accuracy_by_size:
        raise ValueError("No board size data provided for plotting.")

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    color_cycle = plt.cm.get_cmap("tab10")
    # Gather only sizes with non-zero accuracy for plotting
    plotted_sizes_by_label = {}
    y_max = 0

    for idx, (label, sizes) in enumerate(accuracy_by_size.items()):
        filtered = {s: v for s, v in sizes.items() if v > 0}
        if not filtered:
            continue
        sorted_sizes = sorted(filtered.keys())
        values = [filtered[size] * 100 for size in sorted_sizes]
        plotted_sizes_by_label[label] = sorted_sizes
        y_max = max(y_max, max(values))
        ax.plot(
            sorted_sizes,
            values,
            label=label,
            marker="o",
            linewidth=2,
            color=get_bar_color(label, fallback_cmap=color_cycle, index=idx) if label_colors is None else label_colors.get(label, color_cycle(idx % color_cycle.N)),
        )

    # Union of all x values actually plotted
    all_sizes = sorted({s for sizes in plotted_sizes_by_label.values() for s in sizes})
    if all_sizes:
        ax.set_xticks(all_sizes)
        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=10)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")

    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    else:
        ax.set_ylim(0, max(y_max * 1.1, 5))

    # If nothing was plotted, bail out early
    if not plotted_sizes_by_label:
        plt.close(fig)
        raise ValueError("No non-zero accuracy points to plot.")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="0.75", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # legend_cols = min(len(labels), 2) if len(labels) > 1 else 1
        legend_cols = 1
        ax.legend(
            handles,
            labels,
            fontsize=10,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            ncol=legend_cols,
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

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


if __name__ == "__main__":
    bar_names = {
        ("original", "prompt_engineering"): "Default\nBoard",
        # ("coordinate_grid", "prompt_engineering"): "Coordinate\nGrid",
        # ("start_end_marked", "prompt_engineering"): "Start & End\nMarked",
        ("coordinate_grid_and_start_end_marked", "prompt_engineering"): "Coord. Grid\n w. Start &\nEnd Marked",
        ("path_cell_annotated", "prompt_engineering"): "Path Cell\nAnnotated",
        ("text", "prompt_engineering"): "Text\non Board",
        # ("rotated_and_path_cell_annotated", "prompt_engineering"): "Rotated &\nPath Cell\nAnnotated",
    }
    selection_filter = bar_names.keys()

    model = "Qwen3-VL-235B-A22B-Thinking-FP8"
    dataset_subset = "all"
    dataset_split = "test"
    evaluation_dir = f"../evaluation/results/{dataset_split}/{dataset_subset}/{model.split('/')[-1]}"

    individual_results = load_individual_results(evaluation_dir, selection_filter=selection_filter)
    accuracy_by_size = compute_accuracy_by_board_size(individual_results)

    named_accuracy_by_size = {}
    for key, label in bar_names.items():
        if key in accuracy_by_size:
            named_accuracy_by_size[label] = accuracy_by_size[key]

    create_board_size_line_chart(
        named_accuracy_by_size,
        title="", # SPaRC Accuracy by Board Size
        x_label="Board Size (width x height)",
        y_label="Accuracy (%)",
        output_path="images/board_size_accuracy.pdf",
        figsize=(8, 5),
        # y_limit=100,
        label_colors=BAR_LABEL_COLORS,
    )
