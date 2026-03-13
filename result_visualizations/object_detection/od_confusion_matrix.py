import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from objects.tile import get_type_from_abbreviation

ORDERED_ABBREVIATIONS = ["S", "E", "+", "N", "G", ".", "o", "*", "T", "P", "Y", "Missing", "Unknown"]


def parse_eval_key(filename):
    base = filename.replace("_stats_individual.json", "")
    board_part, remainder = base.split("-B_", 1)
    prompt_part = remainder.split("-P_", 1)[0]
    return board_part, prompt_part


def normalize_cell_abbreviation(cell_value, for_prediction=False):
    if cell_value is None:
        return "Missing" if for_prediction else None
    if not isinstance(cell_value, str) or not cell_value:
        return "Unknown" if for_prediction else None

    abbreviation = cell_value[0]
    if abbreviation in {"A", "B", "C", "D"}:
        return "T"
    if abbreviation in {"S", "E", "+", "N", "G", ".", "o", "*", "T", "P", "Y"}:
        return abbreviation
    return "Unknown" if for_prediction else None


def abbreviation_to_label(abbreviation):
    if abbreviation == "Missing":
        return "Missing"
    if abbreviation == "Unknown":
        return "Unknown"
    if abbreviation == "T":
        return "Triangle"
    return get_type_from_abbreviation(abbreviation).replace(" ", "\n")


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


def compute_confusion_counts(puzzles):
    counts = {}

    for puzzle in puzzles:
        target = puzzle.get("puzzle_array", [])
        prediction = puzzle.get("model_solution", [])

        for row_idx, target_row in enumerate(target):
            for col_idx, target_cell in enumerate(target_row):
                true_label = normalize_cell_abbreviation(target_cell, for_prediction=False)
                if true_label is None:
                    continue

                predicted_cell = None
                if row_idx < len(prediction) and col_idx < len(prediction[row_idx]):
                    predicted_cell = prediction[row_idx][col_idx]
                pred_label = normalize_cell_abbreviation(predicted_cell, for_prediction=True)

                row_counts = counts.setdefault(true_label, {})
                row_counts[pred_label] = row_counts.get(pred_label, 0) + 1

    return counts


def build_label_order(confusion_counts):
    true_present = set(confusion_counts.keys())
    pred_present = set()
    for row in confusion_counts.values():
        pred_present.update(row.keys())

    true_labels = [abbr for abbr in ORDERED_ABBREVIATIONS if abbr in true_present]
    true_labels.extend(sorted(abbr for abbr in true_present if abbr not in true_labels))

    pred_labels = [abbr for abbr in ORDERED_ABBREVIATIONS if abbr in pred_present]
    pred_labels.extend(sorted(abbr for abbr in pred_present if abbr not in pred_labels))
    return true_labels, pred_labels


def counts_to_matrix(confusion_counts, true_labels, pred_labels, normalize_rows=True):
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=float)
    true_index = {label: i for i, label in enumerate(true_labels)}
    pred_index = {label: i for i, label in enumerate(pred_labels)}

    for true_label, row in confusion_counts.items():
        if true_label not in true_index:
            continue
        for pred_label, count in row.items():
            if pred_label in pred_index:
                matrix[true_index[true_label], pred_index[pred_label]] += count

    raw_counts = matrix.copy()
    if normalize_rows:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    return matrix, raw_counts


def create_confusion_matrix(
    matrix,
    true_labels,
    pred_labels,
    representation_name,
    output_path="",
    normalize_rows=True,
    figsize=(6.2, 5.6),
    save_format="pdf",
    transparent=False,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    vmax = 1.0 if normalize_rows else max(float(np.max(matrix)), 1.0)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")

    x_axis_labels = [abbreviation_to_label(label) for label in pred_labels]
    y_axis_labels = [abbreviation_to_label(label) for label in true_labels]
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(true_labels)))
    ax.set_xticklabels(x_axis_labels, fontsize=8, fontweight="bold", rotation=45, ha="right")
    ax.set_yticklabels(y_axis_labels, fontsize=8, fontweight="bold")

    ax.set_xlabel("Detected Object", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Object", fontsize=10, fontweight="bold")

    # Draw subtle cell boundaries for readability.
    ax.set_xticks(np.arange(-0.5, len(pred_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(true_labels), 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.35, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = vmax * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value <= 0:
                continue
            label_text = f"{value:.2f}"
            if label_text == "0.00":
                continue
            text_color = "white" if value >= threshold else "black"
            ax.text(j, i, label_text, ha="center", va="center", fontsize=6, color=text_color)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, shrink=0.95)
    colorbar.ax.tick_params(labelsize=8)
    # colorbar_label = "Row-normalized frequency" if normalize_rows else "Cell count"
    colorbar_label = "Frequency"
    colorbar.set_label(colorbar_label, fontsize=10, fontweight="bold")

    representation_single_line = representation_name.replace("\n", " ")
    fig.suptitle(f"SPaRC Object Detection of {representation_single_line} Board", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0,
            format=save_format,
        )

    plt.show()


if __name__ == "__main__":
    selected_representation = ("original", "default")
    selected_representation_name = "Default"
    # selected_representation = ("coordinate_grid_and_start_end_marked", "default")
    # selected_representation_name = "Coord. Grid w. Start & End Marked"
    # selected_representation = ("path_cell_annotated", "default")
    # selected_representation_name = "Path Cell Annotated"
    # selected_representation = ("text", "default")
    # selected_representation_name = "Text on Board"
    # selected_representation = ("coordinate_grid", "default")
    # selected_representation_name = "Coordinate\nGrid"
    # selected_representation = ("start_end_marked", "default")
    # selected_representation_name = "Start & End Marked"
    # selected_representation = ("rotated", "default")
    # selected_representation_name = "Rotated"
    selection_filter = [selected_representation]

    model = "Qwen3-VL-235B-A22B-Thinking-FP8"
    dataset_subset = "all"
    dataset_split = "test"
    project_root = Path(__file__).resolve().parents[2]
    evaluation_dir = project_root / "evaluation" / "results" / "object_detection" / dataset_split / dataset_subset / model.split("/")[-1]

    runs = load_individual_results(evaluation_dir, selection_filter=selection_filter)

    selected_puzzles = runs.get(selected_representation)
    if not selected_puzzles:
        raise ValueError(f"No results found for representation {selected_representation}.")

    confusion_counts = compute_confusion_counts(selected_puzzles)
    true_labels, pred_labels = build_label_order(confusion_counts)
    matrix, _ = counts_to_matrix(confusion_counts, true_labels, pred_labels, normalize_rows=True)

    create_confusion_matrix(
        matrix,
        true_labels,
        pred_labels,
        representation_name=selected_representation_name,
        output_path=str(Path(__file__).resolve().parent / "images" / "od_confusion_matrix.pdf"),
        normalize_rows=True,
    )

