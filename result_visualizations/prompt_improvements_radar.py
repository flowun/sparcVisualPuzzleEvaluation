from util import load_path_analysis_data

from default_radar_chart import create_radar_chart
"""
category_names = {
    ("original", "default_no_tr"): "Image Only w. \nDefault Prompt",
    ("original", "prompt_engineering"): "Image Only w.\nImproved Prompt",
}
"""
category_names = {
    ("original", "default_no_tr"): "Default Prompt",
    ("original", "prompt_engineering"): "Improved Prompt",
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

create_radar_chart(
    named_visualization_data,
    dimensions=dimensions,
    title="Error Types before and after\nPrompt Improvement (lower is better)",
    output_path="images/prompt_improvement_radar.pdf"
)

print(named_visualization_data)