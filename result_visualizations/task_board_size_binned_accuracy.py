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


def compute_accuracy_by_size_bins(individual_results, bin_edges=None):
    if bin_edges is None:
        bin_edges = [25, 49, 81, 121]  # inclusive upper edges for area = width * height
    bin_edges = sorted(bin_edges)

    def bin_label(lower, upper):
        if lower is None:
            return f"≤{upper}"
        if upper is None:
            return f">{lower}"
        return f"{lower + 1}-{upper}"

    accuracy_by_bin = {}
    for key, puzzles in individual_results.items():
        bin_counts = {}
        for puzzle in puzzles:
            width = puzzle.get("width")
            height = puzzle.get("height")
            if width is None or height is None:
                continue
            area = width * height
            lower = None
            upper = None
            for edge in bin_edges:
                if area <= edge:
                    upper = edge
                    break
                lower = edge
            if upper is None:
                lower = bin_edges[-1]
            label = bin_label(lower, upper)
            counts = bin_counts.setdefault(label, {"correct": 0, "total": 0})
            if puzzle.get("is_valid"):
                counts["correct"] += 1
            counts["total"] += 1
        accuracy_by_bin[key] = {
            label: counts["correct"] / counts["total"] if counts["total"] else 0
            for label, counts in bin_counts.items()
        }
    ordered_labels = [bin_label(None, bin_edges[0])]
    for i in range(1, len(bin_edges)):
        ordered_labels.append(bin_label(bin_edges[i - 1], bin_edges[i]))
    ordered_labels.append(bin_label(bin_edges[-1], None))
    return accuracy_by_bin, ordered_labels


def create_binned_chart(
    accuracy_by_bin,
    ordered_bins,
    title="Accuracy by Board Size (binned)",
    x_label="Board Size bin (width x height)",
    y_label="Accuracy (%)",
    output_path="",
    figsize=(8, 5),
    save_format="pdf",
    transparent=False,
    y_limit=None,
    title_fontsize=12,
    label_colors=None,
):
    if not accuracy_by_bin:
        raise ValueError("No board size data provided for plotting.")

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x = range(len(ordered_bins))
    width = 0.1 if len(accuracy_by_bin) > 1 else 0.4
    color_cycle = plt.cm.get_cmap("tab10")
    y_max = 0

    for idx, (label, bins) in enumerate(accuracy_by_bin.items()):
        values = [bins.get(bin_label, 0) * 100 for bin_label in ordered_bins]
        if values:
            y_max = max(y_max, max(values))
        offset = (idx - (len(accuracy_by_bin) - 1) / 2) * width
        ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=label,
            color=get_bar_color(label, fallback_cmap=color_cycle, index=idx) if label_colors is None else label_colors.get(label, color_cycle(idx % color_cycle.N)),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered_bins, fontsize=10, fontweight="bold", rotation=0)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    else:
        ax.set_ylim(0, max(y_max * 1.1, 5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="0.75", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        fig.legend(
            handles,
            labels,
            fontsize=9,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.12),
        )
        fig.subplots_adjust(bottom=0.22)
    else:
        fig.subplots_adjust(bottom=0.12)

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
        ("original", "prompt_engineering"): "Default",
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
    accuracy_by_bin, ordered_bins = compute_accuracy_by_size_bins(individual_results)

    named_accuracy_by_bin = {}
    for key, label in bar_names.items():
        if key in accuracy_by_bin:
            named_accuracy_by_bin[label] = accuracy_by_bin[key]

    create_binned_chart(
        named_accuracy_by_bin,
        ordered_bins,
        title="SPaRC Accuracy by Board Size (binned)",
        x_label="Board Size bin (width x height)",
        y_label="Accuracy (%)",
        output_path="images/board_size_accuracy_binned.pdf",
        figsize=(9, 5),
        label_colors=BAR_LABEL_COLORS,
    )
