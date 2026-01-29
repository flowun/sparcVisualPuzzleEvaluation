import os
import matplotlib.pyplot as plt

from task_board_size_accuracy import load_individual_results


def compute_top_solved_board_size(individual_results):
    """Find the largest solved puzzle (by area) per evaluation key.

    Returns a mapping of eval key -> {
        "area": int,
        "width": int,
        "height": int,
        "task_id": <puzzle identifier or None>
    }
    """
    top_by_key = {}
    for key, puzzles in individual_results.items():
        best = None
        for puzzle in puzzles:
            width = puzzle.get("width")
            height = puzzle.get("height")
            if width is None or height is None:
                continue
            if not puzzle.get("is_valid"):
                continue
            area = width * height
            task_id = puzzle.get("id") or puzzle.get("task_id") or puzzle.get("taskId")
            entry = {"area": area, "width": width, "height": height, "task_id": task_id}
            if best is None or area > best["area"]:
                best = entry
        if best:
            top_by_key[key] = best
    return top_by_key


def find_global_max(top_by_key):
    if not top_by_key:
        return None, None
    return max(top_by_key.items(), key=lambda item: item[1]["area"])


def create_top_solved_bar_chart(top_by_key, label_map, title, output_path, figsize=(8, 5), save_format="pdf", transparent=False):
    entries = []
    for key, label in label_map.items():
        if key in top_by_key:
            entries.append((label, top_by_key[key]))
    if not entries:
        raise ValueError("No solved puzzles found for the selected evaluation types.")

    labels = [lbl for lbl, _ in entries]
    values = [entry["area"] for _, entry in entries]
    annotations = [f"{entry['width']}x{entry['height']}" for _, entry in entries]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    bars = ax.bar(labels, values, color="skyblue", edgecolor="black")
    ax.set_ylabel("Largest Solved Board Size (area)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.1)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="0.75", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for bar, note in zip(bars, annotations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.01, note, ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, transparent=transparent, bbox_inches="tight", pad_inches=0, format=save_format)
    plt.show()


if __name__ == "__main__":
    bar_names = {
        ("original", "prompt_engineering"): "Default",
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
    top_by_key = compute_top_solved_board_size(individual_results)

    named_top = {bar_names[key]: value for key, value in top_by_key.items() if key in bar_names}

    # Print overall largest puzzle information
    global_key, global_entry = find_global_max(top_by_key)
    if global_entry:
        human_label = bar_names.get(global_key, str(global_key))
        print(
            f"Largest solved puzzle: area={global_entry['area']} ({global_entry['width']}x{global_entry['height']}), "
            f"task_id={global_entry.get('task_id')}, solved by={human_label}"
        )
    else:
        print("No solved puzzles found in the selected evaluations.")

    if named_top:
        create_top_solved_bar_chart(
            named_top,
            {name: name for name in named_top.keys()},
            title="Highest Board Size Solved per Evaluation",
            output_path="images/top_solved_board_size.pdf",
            figsize=(8, 5),
        )
    else:
        print("No solved puzzles to plot after applying the selection filter.")
