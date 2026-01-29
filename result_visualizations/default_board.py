from util import load_visualization_data

from default_bar_chart import create_bar_chart

bar_names = {
    ("no_board", "no_board_default"): "Text Only\n(Array)",
    ("original", "default_tr"): "Image & Rule\nCoordinates",
    ("original", "default_no_tr"): "Image Only"
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

print(named_visualization_data)
create_bar_chart(
    named_visualization_data,
    title="SPaRC Accuracy of\nQwen3-VL-235B-Thinking-FP8\nby Board Input Modality",
    x_label="Board Input Modality",
    y_label="Accuracy (%)",
    output_path=f"images/input_modality_motivation.pdf",
    bar_color='skyblue',
    # highlighted_bars=["Image &\nsome Text"],
    figsize=(4, 6/10*9),
)