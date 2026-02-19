import os
import json

name_replacement_dict = {
    "": "",
}

# Shared colors to keep board visualizations consistent.
BOARD_TYPE_COLORS = {
    "original": "#2CA02C",  # green
    "coordinate_grid_and_start_end_marked": "#d62728",  # red
    "path_cell_annotated": "#1f4b99",  # deep blue
    # "text": "#ff69b4",  # pink
    "text": "orange",
    "coordinate_grid": "#9467bd",  # purple
    "start_end_marked": "#8c564b",  # brown
}
BAR_LABEL_COLORS = {
    "Default": BOARD_TYPE_COLORS["original"],
    "Default\nBoard": BOARD_TYPE_COLORS["original"],
    "Coord. Grid\n w. Start &\nEnd Marked": BOARD_TYPE_COLORS["coordinate_grid_and_start_end_marked"],
    "Path Cell\nAnnotated": BOARD_TYPE_COLORS["path_cell_annotated"],
    "Text\non Board": BOARD_TYPE_COLORS["text"],
    "Coordinate\nGrid": BOARD_TYPE_COLORS["coordinate_grid"],
    "Start & End\nMarked": BOARD_TYPE_COLORS["start_end_marked"],
}


def get_bar_color(label, fallback_cmap=None, index=0):
    """Return a consistent color for a label with an optional colormap fallback."""
    if label in BAR_LABEL_COLORS:
        return BAR_LABEL_COLORS[label]
    if fallback_cmap is not None:
        return fallback_cmap(index % fallback_cmap.N)
    return None


def load_visualization_data(evaluation_dir, selection_filter=None, stat='accuracy'):
    """
    Loads evaluation results from JSON files in the specified directory.
    Args:
        evaluation_dir (str): Path to the directory containing evaluation JSON files.
        selection_filter (list, optional): List of tuples (board_type, prompt_type) to filter the results.
            Defaults to None (no filter).
    Returns:
        tuple: A tuple containing:
            - visualization_data (dict): A dictionary with (board_type, prompt_type) keys and accuracy values.
            - board_types (list): A sorted list of unique board types.
            - prompt_types (list): A sorted list of unique prompt types.
        """
    visualization_data = {}
    for file in sorted(os.listdir(evaluation_dir)):
        if file.endswith("_stats_overall.json"):
            overall_stats_file = os.path.join(evaluation_dir, file)
            with open(overall_stats_file, 'r') as f:
                overall_stats = json.load(f)
            board_type = overall_stats['board_type']
            prompt_type = overall_stats['prompt_type']
            if selection_filter is not None and (board_type, prompt_type) not in selection_filter:
                continue
            board_type = name_replacement_dict.get(board_type, board_type)
            prompt_type = name_replacement_dict.get(prompt_type, prompt_type)
            accuracy = overall_stats[stat]
            if (board_type, prompt_type) not in visualization_data:
                visualization_data[(board_type, prompt_type)] = [accuracy]
            else:  # in case there are multiple evaluation results of the same configuration, the average is taken
                visualization_data[(board_type, prompt_type)].append(accuracy)
    for key, accuracies in visualization_data.items():
        visualization_data[key] = sum(accuracies) / len(accuracies)
    if selection_filter is not None:
        sorted_visualization_data = {}
        for key in selection_filter:
            if key in visualization_data:
                sorted_visualization_data[key] = visualization_data[key]
        visualization_data = sorted_visualization_data
        board_types = [key[0] for key in selection_filter]
        prompt_types = [key[1] for key in selection_filter]
    else:
        board_types = sorted(set(key[0] for key in visualization_data.keys()))
        prompt_types = sorted(set(key[1] for key in visualization_data.keys()))
    board_types = [name_replacement_dict.get(bt, bt) for bt in board_types]
    prompt_types = [name_replacement_dict.get(pt, pt) for pt in prompt_types]
    return visualization_data, board_types, prompt_types


def load_path_analysis_data(evaluation_dir, selection_filter=None):
    """
    Loads path analysis results from JSON files in the specified directory.
    Args:
        evaluation_dir (str): Path to the directory containing path analysis JSON files.
        selection_filter (list, optional): List of tuples (board_type, prompt_type) to filter the results.
            Defaults to None (no filter).
    Returns:
        tuple: A tuple containing:
            - path_analysis_data (dict): A dictionary with (board_type, prompt_type) keys and a path analysis dict as values.
            - board_types (list): A sorted list of unique board types.
            - prompt_types (list): A sorted list of unique prompt types.
        """
    path_analysis_data = {}
    for file in sorted(os.listdir(evaluation_dir)):
        if file.endswith("_stats_overall.json"):
            overall_stats_file = os.path.join(evaluation_dir, file)
            with open(overall_stats_file, 'r') as f:
                overall_stats = json.load(f)
            board_type = overall_stats['board_type']
            prompt_type = overall_stats['prompt_type']
            if selection_filter is not None and (board_type, prompt_type) not in selection_filter:
                continue
            board_type = name_replacement_dict.get(board_type, board_type)
            prompt_type = name_replacement_dict.get(prompt_type, prompt_type)
            path_analysis_metrics = overall_stats['avg_path_analysis_metrics']
            if (board_type, prompt_type) not in path_analysis_data:
                path_analysis_data[(board_type, prompt_type)] = [path_analysis_metrics]
            else:  # in case there are multiple evaluation results of the same configuration, the average is taken
                path_analysis_data[(board_type, prompt_type)].append(path_analysis_metrics)
    avg_path_analysis_data = {}
    key_2_replacement_dict = {
        "starts_at_start_ends_at_exit": "Incorrect\nStart/End",
        "fully_valid_path": "Invalid\nPath",
        "no_rule_crossing": "Rule Cell\nCrossing",
        "non_intersecting_line": "Intersecting\nLine",
        "connected_line": "Disconnected\nLine",
    }
    # key_1: (board_type, prompt_type), key_2: metric name
    for key_1 in path_analysis_data.keys():
        avg_path_analysis_data[key_1] = {}
        for key_2 in path_analysis_data[key_1][0].keys():                           # 1 - to change from correctness to error rate
            avg_path_analysis_data[key_1][key_2_replacement_dict.get(key_2, key_2)] = 1 - sum(d[key_2] for d in path_analysis_data[key_1]) / len(path_analysis_data[key_1])
    path_analysis_data = avg_path_analysis_data
    if selection_filter is not None:
        sorted_path_analysis_data = {}
        for key in selection_filter:
            if key in path_analysis_data:
                sorted_path_analysis_data[key] = path_analysis_data[key]
        path_analysis_data = sorted_path_analysis_data
        board_types = [key[0] for key in selection_filter]
        prompt_types = [key[1] for key in selection_filter]
    else:
        board_types = sorted(set(key[0] for key in path_analysis_data.keys()))
        prompt_types = sorted(set(key[1] for key in path_analysis_data.keys()))
    board_types = [name_replacement_dict.get(bt, bt) for bt in board_types]
    prompt_types = [name_replacement_dict.get(pt, pt) for pt in prompt_types]
    return path_analysis_data, board_types, prompt_types
