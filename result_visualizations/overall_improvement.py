from util import load_visualization_data

from default_bar_chart import create_bar_chart

bar_names = {
    ("original", "prompt_engineering"): "Default",
    ("no_board", "no_board_default"): "Text Only\n(no Image)",
    ("coordinate_grid_and_start_end_marked", "prompt_engineering"): "Coord. Grid\n w. Start &\nEnd Marked",
    ("path_cell_annotated", "prompt_engineering"): "Path Cell\nAnnotated",
    ("text", "prompt_engineering"): "Text\non Board",
    ("rotated_and_path_cell_annotated", "prompt_engineering"): "Rotated\nPath Cell\nAnnotatation",
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
    title="SPaRC Accuracy of\nQwen3-VL-235B-Thinking-FP8\nby Puzzle Representation",
    x_label="Board Type",
    y_label="Accuracy (%)",
    output_path=f"images/overall_comparison.pdf",
    bar_color='skyblue',
    highlighted_bars=["Default", "Coord. Grid\n w. Start &\nEnd Marked", "Path Cell\nAnnotated", "Text\non Board"],
    dim_non_highlighted=True,
    figsize=(int(len(bar_names) * 8 / 6), 4),
    y_limit=30,
    title_fontsize=12
)