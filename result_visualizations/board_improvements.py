from util import load_visualization_data

from default_bar_chart import create_bar_chart

bar_names = {
    ("original", "prompt_engineering"): "Default",
    ("coordinate_grid", "prompt_engineering"): "Coordinate\nGrid",
    ("start_end_marked", "prompt_engineering"): "Start & End\nMarked",
    ("coordinate_grid_and_start_end_marked", "prompt_engineering"): "Coord. Grid\n w. Start &\nEnd Marked",
    ("text", "prompt_engineering"): "Text\non Board",
    ("path_cell_annotated", "prompt_engineering"): "Path Cell\nAnnotated",

}
selection_filter = bar_names.keys()

model = "Qwen3-VL-235B-A22B-Thinking-FP8"
dataset_subset = "all"
dataset_split = "test"
evaluation_dir = f"../evaluation/results/{dataset_split}/{dataset_subset}/{model.split('/')[-1]}"

visualization_data, board_types, prompt_types = load_visualization_data(evaluation_dir, selection_filter=selection_filter)

named_visualization_data = {}
for key, value in visualization_data.items():
    named_visualization_data[bar_names[key]] = value

create_bar_chart(
    named_visualization_data,
    title="SPaRC Accuracy of\nQwen3-VL-235B-Thinking-FP8\nby Board Type",
    x_label="Board Type",
    y_label="Accuracy (%)",
    output_path=f"images/board_improvements.pdf",
    bar_color='skyblue',
    # highlighted_bars=["Image &\nsome Text"],
    figsize=(int(len(bar_names) * 8 / 6), 4),
    y_limit=26
)