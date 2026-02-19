from pathlib import Path

from result_visualizations.util import load_visualization_data

from result_visualizations.default_bar_chart import create_bar_chart

bar_names = {
    ("original", "default"): "Default",
    ("coordinate_grid", "default"): "Coordinate\nGrid",
    ("start_end_marked", "default"): "Start & End\nMarked",
    ("coordinate_grid_and_start_end_marked", "default"): "Coord. Grid\n w. Start &\nEnd Marked",
    ("path_cell_annotated", "default"): "Path Cell\nAnnotated",
    ("text", "default"): "Text\non Board",
    ("rotated", "default"): "Rotated\nPath Cell\nAnnotatation",
}
selection_filter = bar_names.keys()

model = "Qwen3-VL-235B-A22B-Thinking-FP8"
dataset_subset = "all"
dataset_split = "test"
project_root = Path(__file__).resolve().parents[2]
evaluation_dir = project_root / "evaluation" / "results" / "object_detection" / dataset_split / dataset_subset / model.split("/")[-1]

visualization_data, board_types, prompt_types = load_visualization_data(evaluation_dir, selection_filter=selection_filter, stat='fraction_average')

named_visualization_data = {}
for key, value in visualization_data.items():
    named_visualization_data[bar_names[key]] = value
print("Named Visualization Data:", named_visualization_data)  # Debugging statement
create_bar_chart(
    named_visualization_data,
    title="SPaRC Object Detection Accuracy of\nQwen3-VL-235B-Thinking-FP8\nby Puzzle Representation",
    x_label="Board Type",
    y_label="Accuracy (%)",
    output_path=str(Path(__file__).resolve().parent / "images" / "od_object_accuracy.pdf"),
    bar_color='skyblue',
    dim_non_highlighted=True,
    figsize=(int(len(bar_names) * 8 / 6), 4),
    y_min=70,
    title_fontsize=12
)