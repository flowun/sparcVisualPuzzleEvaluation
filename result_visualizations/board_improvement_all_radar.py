from util import load_path_analysis_data

from default_radar_chart import create_radar_chart

category_names = {
    ("original", "prompt_engineering"): "Default",
    # ("coordinate_grid", "prompt_engineering"): "Coordinate\nGrid",
    # ("start_end_marked", "prompt_engineering"): "Start & End\nMarked",
    ("coordinate_grid_and_start_end_marked", "prompt_engineering"): "Coord. Grid\n w. Start &\nEnd Marked",
    ("path_cell_annotated", "prompt_engineering"): "Path Cell\nAnnotated",
    # ("text", "prompt_engineering"): "Text\non Board",
}

selection_filter = category_names.keys()

model = "Qwen3-VL-235B-A22B-Thinking-FP8"
dataset_subset = "all"
dataset_split = "test"
evaluation_dir = f"../evaluation/results/{dataset_split}/{dataset_subset}/{model.split('/')[-1]}"

visualization_data, board_types, prompt_types = load_path_analysis_data(evaluation_dir, selection_filter=selection_filter)

dimensions = None
named_visualization_data = {}
for key, value in visualization_data.items():
    named_visualization_data[category_names[key]] = value
    if dimensions is None:
        dimensions = list(value.keys())

from matplotlib import pyplot as plt
create_radar_chart(
    named_visualization_data,
    dimensions=dimensions,
    # title="",
    output_path="images/board_improvement_all_radar.pdf",
)

print(named_visualization_data)