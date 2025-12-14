import os
import asyncio
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime

from sparc.validation import extract_solution_path, validate_solution, analyze_path

from parallel_image_creation import create_board_images_in_parallel
from prompts.payload import create_payload_with_image
from evaluation.request_queue import RequestQueueAsync
import argparse


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
            if 'error' in response:
                print(f"Error in response for ID {data['id']}: {response['error']}")
                pbar.update(1)
                return
            # if response["usage"]["completion_tokens"] == max_tokens:
                # answer not finished properly, make new request asking for the answer
                # new_payload = modify_payload_to_force_final_answer(payload, response)
                # await request_queue.add_request_async(new_payload, make_callback(data, pbar))
                # return

            def _analyze():
                print("\n\nResponse received for ID:", data['id'], "tokens:", response["usage"]["completion_tokens"])
                model_output = response['choices'][0]['message']['content']
                while '####' in model_output and not any(str(n) in model_output.rsplit("####", 1)[-1] for n in range(1, 10)) and not "(" in model_output.rsplit("####", 1)[-1]:
                    # all except everything from last #### (split only once at last occurence)
                    model_output = model_output.rsplit("####", 1)[0]
                path = extract_solution_path(model_output)
                valid = validate_solution(path, data)
                analysis = analyze_path(path, data)
                print("------------------------------------------------\n", response['choices'][0]['message']['content'], "\n\nextracted path:", path, "\nvalid:", valid, "\n------------------------------------------------")
                return path, valid, analysis
            try:
                path, valid, path_analysis = await asyncio.to_thread(_analyze)
                safe_path = path or []
                eval_results.append({
                    "id": data['id'],
                    "extracted_path": safe_path,
                    "is_valid": valid,
                    "path_analysis": path_analysis,
                    "response": response['choices'][0]['message']['content'],
                    "height": data['grid_size']['height'] * 2 + 1,
                    "width": data['grid_size']['width'] * 2 + 1,
                    "difficulty_level": data['difficulty_level'],
                    "difficulty_score": data['difficulty_score'],
                    "polyshapes": data['polyshapes'],
                    "puzzle_array": data['puzzle_array'],
                    "solution_count": data['solution_count'],
                    "solutions": data['solutions'],
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
                else:
                    image_path = None
                json_request = create_payload_with_image(prompt_type, board_type, image_path, data, model, temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k, seed=seed)
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
            "avg_accuracy_by_difficulty_level": (lambda vals: {
                level: sum(1 for r in vals if r['difficulty_level'] == level and r['is_valid']) / max(1, sum(1 for r in vals if r['difficulty_level'] == level))
                for level in range(1, 6)
            })(list(eval_results.values())),
            "avg_path_analysis_metrics": (lambda vals: {
                k: sum(1 for r in vals if (r["path_analysis"][k])) / len(dataset)
                for k in [
                    "starts_at_start_ends_at_exit",
                    "connected_line",
                    "non_intersecting_line",
                    "no_rule_crossing",
                    "fully_valid_path",
                ]
            })(list(eval_results.values())),
            "avg_difficulty_score": sum(r['difficulty_score'] for r in eval_results.values()) / len(dataset),
            "avg_difficulty_level": sum(r['difficulty_level'] for r in eval_results.values()) / len(dataset),
            "avg_path_length": sum(len(r['extracted_path']) for r in eval_results.values()) / len(dataset),
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
                for level in range(1, 6)
            })(list(eval_results.values())),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "max_concurrent_requests": max_concurrent_requests,
            "seed": seed,
        }
        print("Accuracy:", stats["accuracy"], "Avg. Accuracy by Difficulty:", stats["avg_accuracy_by_difficulty_level"])
    except:
        stats = {
            "error": "Failed to compute statistics.",
            "timestamp": timestamp,
        }

    # Saving results
    result_dir = f"evaluation/results/{split}/{subset}/{model.split('/')[1]}"
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

    parser = argparse.ArgumentParser(description="Evaluate SPaRC puzzles with a vision-language model.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", help="Full model name.")
    parser.add_argument("--model-sha", default=None, help="Override model SHA, otherwise resolved via mapping or 'latest'.")
    parser.add_argument("--board-type", default="original", help="Board visualization type.")
    parser.add_argument("--prompt-type", default="default_tr", help="Prompt template type.")
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
