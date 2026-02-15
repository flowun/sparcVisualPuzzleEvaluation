import json
import os
import matplotlib.pyplot as plt
from util import BAR_LABEL_COLORS, get_bar_color


def load_difficulty_data(evaluation_dir, selection_filter=None):
    difficulty_data = {}
    for file in sorted(os.listdir(evaluation_dir)):
        if not file.endswith("_stats_overall.json"):
            continue
        with open(os.path.join(evaluation_dir, file), "r") as f:
            overall_stats = json.load(f)
        board_type = overall_stats["board_type"]
        prompt_type = overall_stats["prompt_type"]
        if selection_filter is not None and (board_type, prompt_type) not in selection_filter:
            continue
        if "avg_accuracy_by_difficulty_level" not in overall_stats:
            continue
        difficulty_levels = {
            int(level): acc for level, acc in overall_stats["avg_accuracy_by_difficulty_level"].items()
        }
        difficulty_data.setdefault((board_type, prompt_type), []).append(difficulty_levels)

    averaged_difficulties = {}
    for key, runs in difficulty_data.items():
        levels = sorted({lvl for run in runs for lvl in run.keys()})
        averaged_difficulties[key] = {
            level: sum(run.get(level, 0) for run in runs) / len(runs) for level in levels
        }
    return averaged_difficulties


def create_difficulty_line_chart(
    difficulty_data,
    title="Difficulty Accuracy",
    x_label="Task Difficulty Level",
    y_label="Accuracy (%)",
    output_path="",
    figsize=(8, 5),
    save_format="pdf",
    transparent=False,
    y_limit=None,
    title_fontsize=12,
    label_colors=None,
):
    if not difficulty_data:
        raise ValueError("No difficulty data provided for plotting.")

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x_levels = sorted({lvl for difficulties in difficulty_data.values() for lvl in difficulties.keys()})
    color_cycle = plt.cm.get_cmap("tab10")
    y_max = 0

    for idx, (label, difficulties) in enumerate(difficulty_data.items()):
        levels = sorted(difficulties.keys())
        values = [difficulties[level] * 100 for level in levels]
        y_max = max(y_max, max(values))
        ax.plot(
            levels,
            values,
            label=label,
            marker="o",
            linewidth=2,
            color=get_bar_color(label, fallback_cmap=color_cycle, index=idx) if label_colors is None else label_colors.get(label, color_cycle(idx % color_cycle.N)),
            alpha=1.0,
        )

    ax.set_xticks(x_levels)
    ax.set_xlabel("Difficulty Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(axis="both", which="major", labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

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
    if handles:
        # legend_cols = min(len(labels), 2)
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

    difficulty_data = load_difficulty_data(evaluation_dir, selection_filter=selection_filter)

    named_difficulty_data = {}
    for key, label in bar_names.items():
        if key in difficulty_data:
            named_difficulty_data[label] = difficulty_data[key]

    create_difficulty_line_chart(
        named_difficulty_data,
        title="SPaRC Accuracy by Task Difficulty",
        x_label="Task Difficulty Level",
        y_label="Accuracy (%)",
        output_path="images/task_difficulty_accuracy.pdf",
        figsize=(8, 5),
        # y_limit=100,
        label_colors=BAR_LABEL_COLORS,
    )
