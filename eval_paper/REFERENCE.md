# Evaluation Reference Guide

This document describes all board representations, prompt types, models, evaluation tasks, worsening conditions, and data formats used in this project. It serves as a self-contained reference for anyone working with the evaluation pipeline or its results.

---

## Table of Contents

1. [Board Representations](#board-representations)
2. [Prompt Types](#prompt-types)
3. [Models](#models)
4. [Evaluation Tasks](#evaluation-tasks)
5. [Worsening Conditions](#worsening-conditions)
6. [Directory Structure](#directory-structure)
7. [File Naming Convention](#file-naming-convention)
8. [Data Formats](#data-formats)
9. [Plotting Configuration](#plotting-configuration)
10. [Coverage Matrix](#coverage-matrix)

---

## Board Representations

Each puzzle board can be rendered in different visual styles. These representations progressively add spatial cues to help VLMs understand the board layout.

| Internal Name | Display Name | Short Label | Description |
|---|---|---|---|
| `original` | Original | Orig. | The default visual representation used in the SPaRC paper. The board is displayed as-is with colored symbols on a grid, without any additional spatial annotations. |
| `start_end_marked` | S/E Markers | S/E | Same as Original, but the **start cell** is marked with an "S" and the **exit cell** with an "E". Motivated by the finding that models often fail to identify the start point. |
| `coordinate_grid` | Axis Labels | Axis | Row and column indices are displayed along the edges of the board (like a spreadsheet), making the coordinate system explicit. |
| `coordinate_grid_and_start_end_marked` | Axis Labels + S/E | Axis+S/E | Combines `coordinate_grid` and `start_end_marked`: axis labels on the edges plus S/E markers on the start and exit cells. |
| `path_cell_annotated` | Cell Coordinates | Cell Coord. | Goes further than axis labels by printing the exact coordinate (e.g., "A3") directly on every cell of the board, removing any ambiguity about cell positions. |
| `text` | Text Symbols | Text Sym. | Same spatial annotations as `path_cell_annotated`, but additionally replaces all graphical rule symbols with text labels (e.g., the star symbol becomes the word "star"), making the board fully interpretable without visual symbol recognition. |

### Representation Hierarchy

```
original
 ├── + S/E markers  →  start_end_marked
 ├── + axis labels  →  coordinate_grid
 │    └── + S/E markers  →  coordinate_grid_and_start_end_marked
 └── + cell coords  →  path_cell_annotated
      └── + text symbols  →  text
```

The ordering used in plots (from most to least aided) is:
`text` > `path_cell_annotated` > `coordinate_grid_and_start_end_marked` > `start_end_marked` > `coordinate_grid` > `original`

---

## Prompt Types

Three prompt variants are used, all tested on the Qwen 3.5 397B model. Other models use only `prompt_engineering`.

| Internal Name | Display Name | Description |
|---|---|---|
| `default_tr` | Default + Text Repr. | The exact prompt used in the original SPaRC paper. Includes both the board screenshot **and** a text representation of the board in the prompt. |
| `default_no_tr` | Default (no Text Repr.) | Same as `default_tr` but with the text representation removed. Only the visual board image is provided. |
| `prompt_engineering` | Prompt Engineering | Our improved prompt version. Uses clearer instructions, structured output format, and is designed to elicit better spatial reasoning from the model. This is the **default prompt** used across all experiments. |

### Object Detection Prompts

For the object detection task, a separate prompt is used:

| Internal Name | Description |
|---|---|
| `default` | The standard prompt for object detection: asks the model to identify and report the coordinates of all rules/symbols on the board. |

---

## Models

### 7 Core Evaluation Models

These are the primary models evaluated across all board types and tasks.

| SPaRC CSV Stem | Test Folder Name | Display Name | Provider | Parameters |
|---|---|---|---|---|
| `google_gemma-3-27b-it` | `gemma-3-27b-it` | Gemma 3 27B | Google | 27B |
| `google_gemma-4-31B-it` | `gemma-4-31B-it` | Gemma 4 31B | Google | 31B |
| `Qwen_Qwen3.5-27B` | `Qwen3.5-27B` | Qwen 3.5 27B | Alibaba/Qwen | 27B |
| `QuantTrio_Qwen3.5-397B-A17B-AWQ` | `Qwen3.5-397B-A17B-AWQ` | Qwen 3.5 397B | Alibaba/Qwen | 397B (17B active, AWQ quantized) |
| `meta-llama_Llama-4-Scout-17B-16E-Instruct` | `Llama-4-Scout-17B-16E-Instruct` | Llama 4 Scout | Meta | 17B per expert, 16 experts (MoE) |
| `mistralai_Mistral-Small-3.2-24B-Instruct-2506` | `Mistral-Small-3.2-24B-Instruct-2506` | Mistral Small 3.2 | Mistral AI | 24B |
| `zai-org_GLM-4.6V` | `GLM-4.6V` | GLM 4.6V | Zhipu AI | — |

### Additional Models (partial coverage)

These models have limited evaluation runs (not all board types):

| Folder Name | Notes |
|---|---|
| `Qwen3-VL-8B-Thinking` | Only `original` board type |
| `Qwen3-VL-30B-A3B-Thinking` | Only `original` board type |
| `Qwen3-VL-32B-Instruct` | `coordinate_grid` and `original` |
| `Qwen3-VL-32B-Thinking` | `path_cell_annotated` only |
| `Qwen3-VL-235B-A22B-Thinking-FP8` | Extended coverage including worsening variants |

### Model Colors

Each model has an assigned color for consistent plotting, derived from the provider's logo. Colors are defined in `plot_config.py` under `MODEL_COLORS`. The color scheme:

- **Qwen** models: purple-indigo shades (`#7C3AED`, `#5B21B6`)
- **Gemma** models: Google blue (`#4285F4`, `#3367D6`)
- **Llama**: Meta blue (`#1565C0`)
- **Mistral**: warm orange (`#D96818`)
- **GLM**: teal-green (`#00695C`)

---

## Evaluation Tasks

### 1. Task Solving (Primary)

The model is given a SPaRC puzzle board image (plus the prompt) and must find a valid path from start to exit that satisfies all rules. This is the main evaluation.

**Location:** `evaluation/results/test/all/<model_folder>/`

**Key metrics:**
- `accuracy` — fraction of puzzles solved correctly (all rules satisfied, valid path)
- `avg_accuracy_by_difficulty_level` — accuracy broken down by difficulty 1–5
- `avg_path_analysis_metrics` — sub-metrics for path validity:
  - `starts_at_start_ends_at_exit` — path correctly connects S to E
  - `connected_line` — path is a continuous connected line
  - `non_intersecting_line` — path does not cross itself
  - `no_rule_crossing` — path does not violate any rules
  - `fully_valid_path` — all of the above are satisfied
- `token_usage` — total prompt, completion, and combined tokens
- `avg_tokens_per_task` — average tokens per puzzle
- `avg_tokens_per_task_by_difficulty_level` — token usage by difficulty

### 2. Object Detection (OD)

The model is given a board image and must report the coordinates of all rule symbols. This tests spatial perception independently from puzzle solving.

**Location:** `evaluation/results/object_detection/test/all/<model_folder>/`

**Key metrics:**
- `accuracy` — fraction of boards where **all** rules are correctly identified (exact match)
- `fraction_average` — average per-rule detection accuracy (more lenient than `accuracy`)
- `accuracy_by_type` — per-symbol-type detection accuracy (e.g., `"Y": 0.65`, `"*": 0.92`)

### 3. SPaRC Baseline

The original SPaRC benchmark results. These serve as the baseline for comparison. The SPaRC evaluation uses its own prompt and board format.

**Location:** `evaluation/results/sparc/`

**Key metrics (from `*_vlm_stats.csv`):**
- Correctly Solved (count and %)
- Fully Valid Paths, Connected Paths, Correct Start/End, etc.
- Per-difficulty breakdown (Difficulty 1–5)

---

## Worsening Conditions

To test robustness, puzzle boards are degraded in three ways. These experiments are available for **Qwen 3.5 397B** and **Gemma 4 31B** (both task solving and object detection).

| Condition | Internal Name | Description |
|---|---|---|
| Low Contrast | `low_contrast` | Board colors are washed out / desaturated, making symbols harder to distinguish visually. |
| Low Resolution | `low_resolution` | Board image is downscaled, reducing pixel-level detail. |
| Rotated | `rotated` | Board image is rotated, disrupting the expected orientation. |

Each worsening condition also has a **recovery variant** that combines the degradation with the `path_cell_annotated` representation to test whether spatial aids can compensate for visual degradation:

| Combined Name | Description |
|---|---|
| `low_contrast_and_path_cell_annotated` | Low contrast board + cell coordinate annotations |
| `low_resolution_and_path_cell_annotated` | Low resolution board + cell coordinate annotations |
| `rotated_and_path_cell_annotated` | Rotated board + cell coordinate annotations |

---

## Directory Structure

```
evaluation/results/
├── sparc/                              # SPaRC baseline results
│   ├── <sparc_csv_stem>_vlm.jsonl      # Raw model outputs
│   ├── <sparc_csv_stem>_vlm_details.csv # Per-puzzle details
│   └── <sparc_csv_stem>_vlm_stats.csv  # Summary statistics
│
├── test/all/                           # Task solving results
│   └── <model_folder>/
│       ├── <board>-B_<prompt>-P_<timestamp>_stats_overall.json
│       └── <board>-B_<prompt>-P_<timestamp>_stats_individual.json
│
├── object_detection/test/all/          # Object detection results
│   └── <model_folder>/
│       ├── <board>-B_<prompt>-P_<timestamp>_stats_overall.json
│       └── <board>-B_<prompt>-P_<timestamp>_stats_individual.json
│
└── figures/                            # Generated plot PDFs
    ├── board_comparison.pdf
    ├── difficulty_comparison.pdf
    ├── heatmap_overview.pdf
    └── ...
```

---

## File Naming Convention

Result files follow a strict naming pattern:

```
<board_type>-B_<prompt_type>-P_<timestamp>_stats_<scope>.json
```

| Component | Description | Example |
|---|---|---|
| `<board_type>` | Board representation used | `original`, `text`, `path_cell_annotated` |
| `-B_` | Separator after board type | — |
| `<prompt_type>` | Prompt variant used | `prompt_engineering`, `default_tr` |
| `-P_` | Separator after prompt type | — |
| `<timestamp>` | Run timestamp (`YYYYMMDD_HHMM`) | `20260401_2159` |
| `_stats_overall` | Aggregated statistics across all puzzles | — |
| `_stats_individual` | Per-puzzle statistics | — |

**Example:**
```
path_cell_annotated-B_prompt_engineering-P_20260402_0915_stats_overall.json
```
This is the overall statistics file for the `path_cell_annotated` board with `prompt_engineering` prompt, run on April 2, 2026 at 09:15.

When multiple runs exist for the same board/prompt combination, the **latest timestamp** is used (files are sorted lexicographically and the last match is selected).

---

## Data Formats

### `stats_overall.json` — Task Solving

```json
{
  "dataset": "lkaesberg/SPaRC",
  "board_type": "original",
  "prompt_type": "prompt_engineering",
  "model": "QuantTrio/Qwen3.5-397B-A17B-AWQ",
  "accuracy": 0.144,
  "avg_accuracy_by_difficulty_level": {
    "1": 0.302, "2": 0.178, "3": 0.149, "4": 0.047, "5": 0.034
  },
  "avg_path_analysis_metrics": {
    "starts_at_start_ends_at_exit": 0.364,
    "connected_line": 0.716,
    "non_intersecting_line": 0.956,
    "no_rule_crossing": 0.948,
    "fully_valid_path": 0.272
  },
  "token_usage": {
    "prompt_tokens": 1021802,
    "completion_tokens": 16148867,
    "total_tokens": 17170669
  },
  "avg_tokens_per_task": {
    "prompt_tokens": 2043.6,
    "completion_tokens": 32297.7,
    "total_tokens": 34341.3
  },
  "avg_tokens_per_task_by_difficulty_level": {
    "1": { "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ... },
    "...": "..."
  },
  "total_requests": 500,
  "temperature": 0.6,
  "max_tokens": 81920,
  "seed": 42
}
```

### `stats_overall.json` — Object Detection

```json
{
  "board_type": "original",
  "prompt_type": "default",
  "model": "QuantTrio/Qwen3.5-397B-A17B-AWQ",
  "accuracy": 0.156,
  "fraction_average": 0.873,
  "accuracy_by_type": {
    "Y": 0.655, "P": 0.821, "T": 0.853, "N": 0.873,
    "G": 0.193, "*": 0.916, "o": 0.900, "+": 0.915,
    "E": 0.452, ".": 0.797, "S": 0.644
  },
  "avg_accuracy_by_difficulty_level": {
    "1": 0.151, "2": 0.136, "3": 0.207, "4": 0.174, "5": 0.101
  }
}
```

### `*_vlm_stats.csv` — SPaRC Baseline

| Metric | Value | Percentage |
|---|---|---|
| Total Puzzles Processed | 500 | 100.0% |
| Correctly Solved | 190 | 38.0% |
| Failed | 310 | 62.0% |
| Fully Valid Paths | 422 | 84.4% |
| Connected Paths | 447 | 89.4% |
| Correct Start/End | 499 | 99.8% |
| Difficulty 1 Solved | 72/86 | 83.7% |
| ... | ... | ... |

---

## Plotting Configuration

All plots are generated using shared settings from `eval_paper/plot_config.py`.

### Figure Dimensions

| Constant | Value | Use Case |
|---|---|---|
| `TEXT_WIDTH_INCHES` | ~6.32 in (455.24 pt) | Full-width figures |
| `COLUMN_WIDTH_INCHES` | ~3.04 in (219.09 pt) | Single-column figures |

### Style Defaults

- **Font:** Times serif with LaTeX rendering (`text.usetex = True`)
- **Base font size:** 8pt (titles: 8pt, axis labels: 8pt, ticks: 7pt, legend: 7pt)
- **DPI:** 300
- **Output format:** PDF (vector, publication-ready)

### Key Functions

| Function | Purpose |
|---|---|
| `setup_plot_style()` | Apply consistent matplotlib style (call at start of every script) |
| `get_model_color(name)` | Look up a model's plot color by display name |
| `get_model_imagebox(name)` | Get a model's logo as a matplotlib `OffsetImage` |
| `add_model_logos(fig, ax, labels)` | Attach logos to y-axis tick labels |
| `get_sparc_accuracy(csv_path)` | Extract overall accuracy from a SPaRC stats CSV |
| `read_json_metric(dir, board, key)` | Read a single metric from a stats JSON |
| `read_json_data(dir, board)` | Read the full JSON dict for a board type |

---

## Coverage Matrix

### Task Solving — Board Types per Model

Checkmark indicates availability. All use `prompt_engineering` unless noted.

| Model | original | start\_end\_marked | coordinate\_grid | coord\_grid + S/E | path\_cell\_annotated | text |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma 3 27B | x | x | x | x | x | x |
| Gemma 4 31B | x | x | x | x | x | x |
| Qwen 3.5 27B | x | x | x | x | x | x |
| Qwen 3.5 397B | x | x | x | x | x | x |
| Llama 4 Scout | x | x | x | x | x | x |
| Mistral Small 3.2 | x | x | x | x | x | x |
| GLM 4.6V | x | x | x | x | x | x |

### Prompt Ablations (Qwen 3.5 397B only)

| Board Type | `default_tr` | `default_no_tr` | `prompt_engineering` |
|---|:---:|:---:|:---:|
| original | x | x | x |
| start\_end\_marked | x | x | x |
| coordinate\_grid | x | x | x |
| coord\_grid + S/E | x | x | x |
| path\_cell\_annotated | x | x | x |
| text | x | x | x |

### Worsening Conditions

| Model | low\_contrast | low\_resolution | rotated | + path\_cell\_annotated recovery |
|---|:---:|:---:|:---:|:---:|
| Qwen 3.5 397B | x | x | x | x (all three) |
| Gemma 4 31B | x | x | x | x (all three) |

### Object Detection — Board Types per Model

| Model | original | start\_end\_marked | coordinate\_grid | coord\_grid + S/E | path\_cell\_annotated | text |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma 3 27B | x | x | x | x | x | x |
| Gemma 4 31B | x | x | x | x | x | x |
| Qwen 3.5 27B | x | x | x | x | x | x |
| Qwen 3.5 397B | x | x | x | x | x | x |
| Llama 4 Scout | x | x | x | x | x | x |
| Mistral Small 3.2 | x | x | x | x | x | x |
| GLM 4.6V | x | x | x | x | x | x |

### Object Detection Worsening (same models as task worsening)

| Model | low\_contrast | low\_resolution | rotated | + path\_cell\_annotated recovery |
|---|:---:|:---:|:---:|:---:|
| Qwen 3.5 397B | x | x | x | x (all three) |
| Gemma 4 31B | x | x | x | x (all three) |
