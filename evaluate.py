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


def evaluate(model, model_sha="latest", split="test", subset="all", board_type="original", prompt_type="default", api_port=8000, max_concurrent_requests=5, temperature=0.6, max_tokens=10000, top_p=0.95, top_k=20):
    start_time = time.time()
    # preparation
    dataset_revision = "195579019ab44fce4f394bb03af04bf598956e4b"
    dataset = load_dataset("lkaesberg/SPaRC", subset, split=split, revision=dataset_revision)
    board_visualization_dir = f"data/boards/{board_type}/{split}/{subset}"
    if not os.path.exists(board_visualization_dir):
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
                image_path = os.path.join(board_visualization_dir, data['id'] + ".png")
                json_request = create_payload_with_image(prompt_type, board_type, image_path, data, model, temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)
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
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": "d464a056915e088a7621533813ed553ceea73a6e",
    }
    model = "Qwen/Qwen3-VL-32B-Thinking"
    port = 8002
    evaluate(
        model=model,
        model_sha=model_version_dict.get(model, "latest"),
        split="test",
        subset="all",
        board_type="original",
        prompt_type="prompt_engineering",
        api_port=port,
        temperature=0.6,
        max_tokens=81920,
        max_concurrent_requests=200,
    )

# with 200 concurrent requests:
# Qwen/Qwen3-VL-32B-Thinking on 4x A100 40GB (80GB similar): 3200 tokens/s at the start, 1000 tokens/s later when 60 batches are running
# 1/10 less batch size --> approx. 10% less throughput