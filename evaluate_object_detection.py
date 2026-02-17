import os
import asyncio
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
import re
import ast
import warnings

from parallel_image_creation import create_board_images_in_parallel, create_board_image
from prompts.payload import create_payload_with_image
from evaluation.request_queue import RequestQueueAsync
import argparse


def extract_model_solution(model_output):
    """
    Extract a 2D array representation (list of lists of strings) from model output.
    Tries to be resilient to code fences, extra text, and spacing/newlines.
    """
    def _warn_and_empty(reason):
        warnings.warn(f"Could not parse model solution: {reason}", UserWarning)
        return [[]]

    if model_output is None:
        return _warn_and_empty("model_output is None")

    solution_marker = "####"
    solution_part = model_output.split(solution_marker)[-1] if solution_marker in model_output else model_output

    # Strip markdown code fences and language hints
    solution_part = re.sub(r"```[\w-]*", " ", solution_part)
    solution_part = solution_part.replace("```", " ")

    # Strip '\n', '\' and ' '
    solution_part = solution_part.replace("\\n", "").replace("\\", "").replace(" ", "")

    # Keep only the first complete bracketed expression
    start = solution_part.find("[")
    end = solution_part.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return _warn_and_empty("no bracketed array found")
    bracketed = solution_part[start : end + 1]

    # Normalize whitespace
    bracketed = re.sub(r"\s+", " ", bracketed).strip()

    def _literal_parse(text):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and all(isinstance(r, list) for r in parsed):
                # Ensure all entries are strings
                return [[str(c).strip() for c in row] for row in parsed]
        except Exception:
            return None
        return None

    parsed = _literal_parse(bracketed)
    if parsed is not None:
        return parsed

    # Fallback: regex row extraction
    rows = []
    for row_text in re.findall(r"\[([^\[\]]+)\]", bracketed):
        tokens = []
        for g1, g2, g3 in re.findall(r'"([^"]+)"|\'([^\']+)\'|([^\s,]+)', row_text):
            token = g1 or g2 or g3
            if token:
                tokens.append(token.strip())
        if tokens:
            rows.append(tokens)

    if not rows:
        return _warn_and_empty("parsed rows were empty")

    return rows

def is_same_poly(provided_cell, correct_cell, poly_definitions):
    """
    Compare polyshape string of model output with correct solution
    :param provided_cell: string in format like P-B-110-110-010
    :param correct_cell: string in format like P-B-16 (id is binary representation of shape, see poly_definitions)
    :param poly_definitions: dictionary mapping shape ids to array, e.g. {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
    :return: True if the provided cell matches the correct cell, False otherwise
    """
    provided_parts = provided_cell.split("-")
    correct_parts = correct_cell.split("-")
    if not (provided_cell.startswith("P-") or provided_cell.startswith("Y-")):
        raise ValueError(f"Invalid polyshape format: {provided_cell}")
    if len(provided_parts) < 3:
        return False  # not enough parts to define a full polyshape
    if len(provided_parts) > 25:
        return False  # too big shape, cannot be correct
    if not provided_parts[0] == correct_parts[0]:
        return False  # False object type
    if not provided_parts[1] == correct_parts[1]:
        return False  # False color
    correct_shape = poly_definitions[correct_parts[2]]
    # remove rows/columns from the correct list with only 0s
    keep_i, keep_j = set(), set()
    for i, row in enumerate(correct_shape):
        for j, cell in enumerate(row):
            if cell == 1:
                keep_i.add(i)
                keep_j.add(j)
    if not keep_i or not keep_j:
        return False
    correct_shape = [[correct_shape[i][j] for j in sorted(keep_j)] for i in sorted(keep_i)]

    provided_rows = provided_parts[2:]
    if not provided_rows:
        return False
    provided_shape = []
    row_lengths = set()
    for row in provided_rows:
        if not row:
            return False
        parsed_row = []
        for ch in row:
            if ch == "1":
                parsed_row.append(1)
            elif ch == "0":
                parsed_row.append(0)
            else:
                return False  # invalid character
        provided_shape.append(parsed_row)
        row_lengths.add(len(parsed_row))
    if len(row_lengths) != 1:
        return False  # inconsistent row lengths

    keep_i_provided, keep_j_provided = set(), set()
    for i, row in enumerate(provided_shape):
        for j, cell in enumerate(row):
            if cell == 1:
                keep_i_provided.add(i)
                keep_j_provided.add(j)
    if not keep_i_provided or not keep_j_provided:
        return False
    provided_shape = [[provided_shape[i][j] for j in sorted(keep_j_provided)] for i in sorted(keep_i_provided)]
    return provided_shape == correct_shape

def analyze_solution(model_solution, data):
    """
    Analyze whether the provided solution is fully valid and what fraction of the cells are correct
    :param model_solution: list of lists of strings representing the model's extracted solution array
    :param data: dictionary containing the puzzle data, including the correct solution array and polyshape definitions
    :return tuple (is_fully_valid: bool, valid_fraction: float, per_type_stats: dict)
    """
    if model_solution is None:
        return 0, 0, {}
    if not isinstance(model_solution, list) or not all(isinstance(row, list) for row in model_solution):
        raise ValueError("Model solution must be a list of lists")

    n_valid, n_total = 0, 0
    per_type_counts = {}  # {type: {"total": int, "correct": int}}
    for i in range(len(data["puzzle_array"])):
        for j in range(len(data["puzzle_array"][0])):
            correct_cell = data["puzzle_array"][i][j]
            provided_cell = model_solution[i][j] if i < len(model_solution) and j < len(model_solution[i]) else None
            cell_type = correct_cell[0] if correct_cell else None
            if cell_type in ["A", "B", "C", "D"]:
                cell_type = "T"
            if cell_type is not None:
                per_type_counts.setdefault(cell_type, {"total": 0, "correct": 0})
                per_type_counts[cell_type]["total"] += 1
            n_total += 1

            if provided_cell is None:
                continue

            is_correct = False
            if correct_cell.startswith(("P-", "Y-")):
                if provided_cell.startswith(("P-", "Y-")) and is_same_poly(provided_cell, correct_cell, json.loads(data["polyshapes"])):
                    is_correct = True
            else:
                if provided_cell == correct_cell:
                    is_correct = True

            if is_correct:
                n_valid += 1
                if cell_type is not None:
                    per_type_counts[cell_type]["correct"] += 1

    valid_fraction = n_valid / n_total if n_total > 0 else 0

    per_type_stats = {
        t: {
            "total": v["total"],
            "correct": v["correct"],
            "fraction": (v["correct"] / v["total"]) if v["total"] > 0 else -1,
        }
        for t, v in per_type_counts.items()
    }

    return 1 if n_valid == n_total else 0, valid_fraction, per_type_stats

def evaluate(model, model_sha="latest", split="test", subset="all", board_type="original", prompt_type="default", api_port=8000, max_concurrent_requests=5, temperature=0.6, max_tokens=10000, top_p=0.95, top_k=20, seed=42):
    start_time = time.time()
    # preparation
    dataset_revision = "195579019ab44fce4f394bb03af04bf598956e4b"
    dataset = load_dataset("lkaesberg/SPaRC", subset, split=split, revision=dataset_revision)
    board_visualization_dir = f"data/boards/{board_type}/{split}/{subset}"
    if board_type == "no_board" and prompt_type == "no_board_default":  # sparc evaluation without images and without specifying prompts (as in initial SPaRC paper)
        board_visualization_dir = None
    if board_visualization_dir is not None and not os.path.exists(board_visualization_dir):
        print(f"Creating board images in {board_visualization_dir}...")
        create_board_images_in_parallel(dataset, split_savename=split, subset_savename=subset, plot_type=board_type)

    eval_results = []

    request_queue = RequestQueueAsync(
        max_concurrent_requests=max_concurrent_requests,
        api_url=f"http://127.0.0.1:{api_port}/v1/chat/completions"
    )

    def make_callback(data, pbar):
        async def on_response(response, payload):
            # print("Response is:", response)
            if 'error' in response:
                print(f"Error in response for ID {data['id']}: {response['error']}")
                pbar.update(1)
                return

            def _analyze():
                # print("\n\nResponse received for ID:", data['id'], "tokens:", response["usage"]["completion_tokens"])
                model_output = response['choices'][0]['message']['content']
                while '####' in model_output and not all(c in model_output.rsplit("####", 1)[-1] for c in ['[', ']', ',']):
                    # all except everything from last #### (split only once at last occurence)
                    model_output = model_output.rsplit("####", 1)[0]
                model_solution = extract_model_solution(model_output)
                print(response['choices'][0]['message']['content'], "\n\nmodel solution:", model_solution)
                fully_valid, valid_fraction, per_type_stats = analyze_solution(model_solution, data)
                print("\nfully_valid:", fully_valid, "\nvalid_fraction:", valid_fraction, "\n------------------------------------------------")
                return fully_valid, valid_fraction, per_type_stats, model_solution
            try:
                fully_valid, valid_fraction, per_type_stats, model_solution = await asyncio.to_thread(_analyze)
                safe_solution = model_solution or []
                eval_results.append({
                    "id": data['id'],
                    "model_solution": safe_solution,
                    "is_valid": fully_valid,
                    "valid_fraction": valid_fraction,
                    "analysis": per_type_stats,
                    "response": response['choices'][0]['message']['content'],
                    "height": data['grid_size']['height'] * 2 + 1,
                    "width": data['grid_size']['width'] * 2 + 1,
                    "difficulty_level": data['difficulty_level'],
                    "difficulty_score": data['difficulty_score'],
                    "polyshapes": data['polyshapes'],
                    "puzzle_array": data['puzzle_array'],
                    "token_usage": response["usage"],
                })
            except Exception as e:
                print(f"Exception during analysis for ID {data['id']}: {e}")
            finally:
                pbar.update(1)
        return on_response

    async def _run():
        await request_queue.start()

        # enqueue work
        with tqdm(total=len(dataset), desc="Evaluating") as pbar:
            for data in dataset:
                if board_visualization_dir is not None:
                    image_path = os.path.join(board_visualization_dir, data['id'] + ".png")
                    if not os.path.exists(image_path):
                        # Generate missing image on demand to avoid FileNotFoundError.
                        create_board_image(data, split_savename=split, subset_savename=subset, plot_type=board_type)
                else:
                    image_path = None
                json_request = create_payload_with_image(prompt_type, board_type, image_path, data, model, temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k, seed=seed, object_detection_ablation=True)
                await request_queue.add_request_async(json_request, make_callback(data, pbar))

            await request_queue.queue.join()  # wait until all tasks are processed
        await request_queue.close()

    asyncio.run(_run())
    evaluation_duration_seconds = time.time() - start_time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Process results
    print(f"Processing and saving results...")
    eval_results = {r['id']: r for r in eval_results}
    try:
        difficulty_levels = []
        for i in range(1, 6):
            for r in eval_results.values():
                if r['difficulty_level'] == i:
                    difficulty_levels.append(i)
                    break
        stats = {
            "dataset": "lkaesberg/SPaRC",
            "dataset_revision": dataset_revision,
            "dataset_subset": subset,
            "dataset_split": split,
            "board_type": board_type,
            "prompt_type": prompt_type,
            "model": model,
            "model_sha": model_sha,
            "accuracy": sum(1 for r in eval_results.values() if r['is_valid']) / len(dataset),
            "fraction_average": sum(r['valid_fraction'] for r in eval_results.values()) / len(dataset),
            "accuracy_by_type": {t: (sum(r["analysis"].get(t, {}).get("correct", 0) for r in eval_results.values()) / sum(r["analysis"].get(t, {}).get("total", 0) for r in eval_results.values())) if sum(r["analysis"].get(t, {}).get("total", 0) for r in eval_results.values()) > 0 else 0.0 for t in {k for r in eval_results.values() for k in r["analysis"]}},
            "avg_accuracy_by_difficulty_level": (lambda vals: {
                level: sum(1 for r in vals if r['difficulty_level'] == level and r['is_valid']) / max(1, sum(1 for r in vals if r['difficulty_level'] == level))
                for level in difficulty_levels
            })(list(eval_results.values())),
            "avg_difficulty_score": sum(r['difficulty_score'] for r in eval_results.values()) / len(dataset),
            "avg_difficulty_level": sum(r['difficulty_level'] for r in eval_results.values()) / len(dataset),
            "total_requests": len(dataset),
            "timestamp": timestamp,
            "evaluation_duration_seconds": evaluation_duration_seconds,
            "token_usage": {
                "prompt_tokens": sum(r['token_usage']['prompt_tokens'] for r in eval_results.values()),
                "completion_tokens": sum(r['token_usage']['completion_tokens'] for r in eval_results.values()),
                "total_tokens": sum(r['token_usage']['total_tokens'] for r in eval_results.values()),
            },
            "avg_tokens_per_task": {
                "prompt_tokens": sum(r['token_usage']['prompt_tokens'] for r in eval_results.values()) / len(dataset),
                "completion_tokens": sum(r['token_usage']['completion_tokens']for r in eval_results.values()) / len(dataset),
                "total_tokens": sum(r['token_usage']['total_tokens'] for r in eval_results.values()) / len(dataset),
            },
            "avg_tokens_per_task_by_difficulty_level": (lambda vals: {
                level: {
                    "prompt_tokens": sum(r['token_usage']['prompt_tokens'] for r in vals if r['difficulty_level'] == level) / sum(1 for r in vals if r['difficulty_level'] == level),
                    "completion_tokens": sum(r['token_usage']['completion_tokens'] for r in vals if r['difficulty_level'] == level) / sum(1 for r in vals if r['difficulty_level'] == level),
                    "total_tokens": sum(r['token_usage']['total_tokens'] for r in vals if r['difficulty_level'] == level) / sum(1 for r in vals if r['difficulty_level'] == level),
                }
                for level in difficulty_levels
            })(list(eval_results.values())),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "max_concurrent_requests": max_concurrent_requests,
            "seed": seed,
        }
        print("Accuracy:", stats["accuracy"], "Fraction average:", stats["fraction_average"], "Avg. Accuracy by Difficulty:", stats["avg_accuracy_by_difficulty_level"])
    except:
        stats = {
            "error": "Failed to compute statistics.",
            "timestamp": timestamp,
        }

    # Saving results
    result_dir = f"evaluation/results/object_detection/{split}/{subset}/{model.split('/')[1]}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    stats_filepath = os.path.join(result_dir, f"{board_type}-B_{prompt_type}-P_{timestamp}_stats_overall.json")
    model_output_filepath = os.path.join(result_dir, f"{board_type}-B_{prompt_type}-P_{timestamp}_stats_individual.json")

    with open(stats_filepath, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(model_output_filepath, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    model_version_dict = {
        "Qwen/Qwen3-VL-8B-Thinking": "41ea130ce6eaaf7829c72dfc0e4597d49741ed18",
        "Qwen/Qwen3-VL-32B-Instruct": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        "Qwen/Qwen3-VL-32B-Thinking": "7edd10ffd1196091948fb245ff63e406ccb2d4d1",
        "Qwen/Qwen3-VL-30B-A3B-Thinking": "7e9bbfa2c1b2059edd18160793fd421194da2c10",
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": "d464a056915e088a7621533813ed553ceea73a6e",
        "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8": "c6c469b4fb011e422f962f98b457743dbd6e7052"
    }

    parser = argparse.ArgumentParser(description="Evaluate SPaRC object detection with a vision-language model.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8", help="Full model name.")
    parser.add_argument("--model-sha", default=None, help="Override model SHA, otherwise resolved via mapping or 'latest'.")
    parser.add_argument("--board-type", default="original", help="Board visualization type.")
    parser.add_argument("--prompt-type", default="default", help="Prompt template type.")
    parser.add_argument("--subset", default="all", help="Dataset subset.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--api-port", type=int, default=8000, help="Local API port.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=81920, help="Max completion tokens.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--max-concurrent-requests", type=int, default=200, help="Concurrency limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    args = parser.parse_args()

    resolved_model_sha = args.model_sha or model_version_dict.get(args.model, "latest")

    evaluate(
        model=args.model,
        model_sha=resolved_model_sha,
        split=args.split,
        subset=args.subset,
        board_type=args.board_type,
        prompt_type=args.prompt_type,
        api_port=args.api_port,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        max_concurrent_requests=args.max_concurrent_requests,
        seed=args.seed,
    )
