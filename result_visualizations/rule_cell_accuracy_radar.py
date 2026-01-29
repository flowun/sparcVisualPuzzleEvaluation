import json
import os
import matplotlib.pyplot as plt

from default_radar_chart import create_radar_chart
from util import BAR_LABEL_COLORS


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


def categorize_rule_cell(abbr, include_gap=True):
    base = abbr.split("-", 1)[0]
    if not base:
        return None
    first = base[0]
    if first in {"A", "B", "C", "D"}:
        return "Triangle"
    if first == "P":
        return "Positive Polyshape"
    if first == "Y":
        return "Negative Polyshape"
    if first == "o":
        return "Stone"
    if first == "*":
        return "Star"
    if first == "N":
        return "Square"
    if include_gap and first == "G":
        return "Gap"
    return None


def extract_rule_categories(puzzle_array, include_gap=True):
    categories = set()
    for row in puzzle_array:
        for cell in row:
            cat = categorize_rule_cell(cell, include_gap=include_gap)
            if cat:
                categories.add(cat)
    return categories


def compute_accuracy_by_rule_type(individual_results, include_rules=None, include_gap=True):
    accuracy_by_rule = {}
    for key, puzzles in individual_results.items():
        rule_counts = {}
        for puzzle in puzzles:
            puzzle_categories = extract_rule_categories(puzzle.get("puzzle_array", []), include_gap=include_gap)
            for category in puzzle_categories:
                if include_rules is not None and category not in include_rules:
                    continue
                counts = rule_counts.setdefault(category, {"correct": 0, "total": 0})
                if puzzle.get("is_valid"):
                    counts["correct"] += 1
                counts["total"] += 1
        accuracy_by_rule[key] = {
            category: counts["correct"] / counts["total"] if counts["total"] else 0
            for category, counts in rule_counts.items()
        }
    return accuracy_by_rule


def build_dimensions(accuracy_by_rule, include_rules=None):
    ordered = [
        "Square",
        "Triangle",
        "Positive Polyshape",
        "Negative Polyshape",
        "Stone",
        "Star",
        "Gap",
    ]
    present = {cat for values in accuracy_by_rule.values() for cat in values.keys()}
    if include_rules is not None:
        present &= set(include_rules)
    dimensions = [c for c in ordered if c in present]
    for cat in sorted(present):
        if cat not in dimensions:
            dimensions.append(cat)
    return dimensions


def prepare_radar_data(accuracy_by_rule, dimensions):
    radar_data = {}
    for label, rules in accuracy_by_rule.items():
        radar_data[label] = [rules.get(dim, 0) for dim in dimensions]
    return radar_data


if __name__ == "__main__":
    v = 2
    if v == 1:
        bar_names = {
            ("original", "prompt_engineering"): "Default\nBoard",
            ("coordinate_grid", "prompt_engineering"): "Coordinate\nGrid",
            ("start_end_marked", "prompt_engineering"): "Start & End\nMarked",
            # ("coordinate_grid_and_start_end_marked", "prompt_engineering"): "Coord. Grid\n w. Start &\nEnd Marked",
            # ("path_cell_annotated", "prompt_engineering"): "Path Cell\nAnnotated",
            # ("text", "prompt_engineering"): "Text\non Board",
            # ("rotated_and_path_cell_annotated", "prompt_engineering"): "Rotated &\nPath Cell\nAnnotated",
        }
    else:
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

    include_gap = True
    include_rules = None  # e.g., ["Square", "Triangle", "Positive Polyshape", "Negative Polyshape", "Stone", "Star", "Gap"]

    model = "Qwen3-VL-235B-A22B-Thinking-FP8"
    dataset_subset = "all"
    dataset_split = "test"
    evaluation_dir = f"../evaluation/results/{dataset_split}/{dataset_subset}/{model.split('/')[-1]}"

    individual_results = load_individual_results(evaluation_dir, selection_filter=selection_filter)
    accuracy_by_rule = compute_accuracy_by_rule_type(
        individual_results,
        include_rules=include_rules,
        include_gap=include_gap,
    )

    named_accuracy_by_rule = {}
    color_cycle = plt.cm.get_cmap("tab10")
    color_list = []
    for key, label in bar_names.items():
        if key in accuracy_by_rule:
            named_accuracy_by_rule[label] = accuracy_by_rule[key]
            color_list.append(BAR_LABEL_COLORS.get(label, color_cycle(len(color_list) % color_cycle.N)))

    dimensions = build_dimensions(named_accuracy_by_rule, include_rules=include_rules)
    radar_data = prepare_radar_data(named_accuracy_by_rule, dimensions)

    create_radar_chart(
        radar_data,
        dimensions=dimensions,
        title="",  # SPaRC Accuracy by Rule Cell Type
        figsize=(7.5, 7.5),
        rlabel_format="percent",
        rmax=0.08 if v == 1 else 0.3,
        rticks=(0.02, 0.04, 0.06, 0.08) if v == 1 else (0.1, 0.2, 0.3),
        start_angle_deg=90,
        clockwise=False,
        legend=True,
        output_path=f"images/rule_cell_accuracy_radar_v{v}.pdf",
        save_format="pdf",
        transparent=False,
        fill=True,
        fill_alpha=0.08,
        title_fontsize=16,
        legend_fontsize=12,
        dimension_fontsize=12,
        tick_fontsize=9,
        colors=color_list,
        reverse_plotting=True,
    )
