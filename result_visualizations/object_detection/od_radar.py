from pathlib import Path
import json
import os

from objects.tile import get_type_from_abbreviation
from result_visualizations.default_radar_chart import create_radar_chart

bar_names = {
    ("original", "default"): "Default",
    # ("coordinate_grid", "default"): "Coordinate\nGrid",
    # ("start_end_marked", "default"): "Start & End\nMarked",
    ("coordinate_grid_and_start_end_marked", "default"): "Coord. Grid\n w. Start &\nEnd Marked",
    ("path_cell_annotated", "default"): "Path Cell\nAnnotated",
    ("text", "default"): "Text\non Board",
    # ("rotated", "default"): "Rotated\nPath Cell\nAnnotatation",
}
selection_filter = bar_names.keys()

ORDERED_ABBREVIATIONS = ["S", "E", "+", "N", "G", ".", "o", "*", "T", "P", "Y"]


def normalize_abbreviation(abbreviation):
    return "A" if abbreviation == "T" else abbreviation


def format_type_label(type_name):
    return type_name.replace(" ", "\n")


def build_dimensions(available_abbreviations):
    ordered = []
    for abbreviation in ORDERED_ABBREVIATIONS:
        normalized = normalize_abbreviation(abbreviation)
        if normalized in available_abbreviations and normalized not in ordered:
            ordered.append(normalized)
    ordered.extend(sorted(a for a in available_abbreviations if a not in ordered))
    return [format_type_label(get_type_from_abbreviation(a)) for a in ordered]


def load_accuracy_by_type(evaluation_dir, selection_filter=None):
    visualization_data = {}
    available_abbreviations = set()

    for file in sorted(os.listdir(evaluation_dir)):
        if not file.endswith("_stats_overall.json"):
            continue
        overall_stats_file = os.path.join(evaluation_dir, file)
        with open(overall_stats_file, "r") as f:
            overall_stats = json.load(f)

        board_type = overall_stats["board_type"]
        prompt_type = overall_stats["prompt_type"]
        if selection_filter is not None and (board_type, prompt_type) not in selection_filter:
            continue

        accuracy_by_type = overall_stats.get("accuracy_by_type", {})
        normalized = {}
        counts = {}
        for abbreviation, value in accuracy_by_type.items():
            normalized_abbreviation = normalize_abbreviation(abbreviation)
            available_abbreviations.add(normalized_abbreviation)
            normalized[normalized_abbreviation] = normalized.get(normalized_abbreviation, 0.0) + value
            counts[normalized_abbreviation] = counts.get(normalized_abbreviation, 0) + 1

        named_accuracy = {}
        for abbreviation, total in normalized.items():
            type_name = get_type_from_abbreviation(abbreviation)
            named_accuracy[format_type_label(type_name)] = total / counts[abbreviation]

        visualization_data[(board_type, prompt_type)] = named_accuracy

    dimensions = build_dimensions(available_abbreviations)
    return visualization_data, dimensions


model = "Qwen3-VL-235B-A22B-Thinking-FP8"
dataset_subset = "all"
dataset_split = "test"
project_root = Path(__file__).resolve().parents[2]
evaluation_dir = project_root / "evaluation" / "results" / "object_detection" / dataset_split / dataset_subset / model.split("/")[-1]

visualization_data, dimensions = load_accuracy_by_type(evaluation_dir, selection_filter=selection_filter)

named_visualization_data = {}
for key in bar_names:
    if key in visualization_data:
        named_visualization_data[bar_names[key]] = visualization_data[key]

create_radar_chart(
    named_visualization_data,
    dimensions=dimensions,
    title="SPaRC Object Detection Accuracy\nby Object Type of\nQwen3-VL-235B-Thinking-FP8",
    figsize=(8, 8),
    rlabel_format="percent",
    start_angle_deg=45,
    clockwise=False,
    legend=True,
    reverse_plotting=True,
    dimension_label_pad=16,
    output_path=str(Path(__file__).resolve().parent / "images" / "od_radar.pdf"),
    save_format="pdf",
    transparent=False,
    fill=True,
)
