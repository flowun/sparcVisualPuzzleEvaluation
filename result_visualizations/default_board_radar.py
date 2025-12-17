from util import load_path_analysis_data

from default_radar_chart import create_radar_chart

category_names = {
    # ("no_board", "no_board_default"): "Text Only",
    # ("original", "default_tr"): "Image &\nsome Text",
    ("original", "default_no_tr"): "Image Only"
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
    # title="Error Types by Input\nModality (lower is better)",
    # title="Error Types for Image Only\nBoard Input (lower is better)",
    output_path="images/input_modality_motivation_radar.pdf",
    # colors=[plt.cm.tab10.colors[4], plt.cm.tab10.colors[9], plt.cm.tab10.colors[2]]
    show_single_legend=True,
)

print(named_visualization_data)