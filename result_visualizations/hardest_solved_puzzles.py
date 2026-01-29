import json
import os
from typing import Dict, Iterable, List, Set, Tuple


MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking-FP8"
RESULTS_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "evaluation", "results")
)
MODEL_RESULTS_DIR = os.path.join(RESULTS_BASE, "test", "all", MODEL_NAME)


def parse_eval_key(filename: str) -> Tuple[str, str]:
    """Extract board_type and prompt_type from an evaluation filename."""
    base = filename.replace("_stats_individual.json", "")
    board_part, remainder = base.split("-B_", 1)
    prompt_part = remainder.split("-P_", 1)[0]
    return board_part, prompt_part


def iter_individual_eval_files(base_dir: str) -> Iterable[str]:
    """Yield paths to all *_stats_individual.json files under base_dir."""
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith("_stats_individual.json"):
                yield os.path.join(root, filename)


def load_entries(file_path: str) -> Iterable[Dict]:
    """Load a stats_individual file and yield its puzzle entries."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.values()
    if isinstance(data, list):
        return data
    return []


def collect_solved_puzzles(base_dir: str) -> Dict[str, Dict[str, object]]:
    """
    Return mapping of puzzle id -> {score: float, reps: set[(board, prompt)]}
    for all solved puzzles.
    """
    solved: Dict[str, Dict[str, object]] = {}
    for file_path in iter_individual_eval_files(base_dir):
        fname = os.path.basename(file_path)
        board_type, prompt_type = parse_eval_key(fname)
        try:
            entries = load_entries(file_path)
        except Exception as exc:
            print(f"Skipping {file_path} ({exc})")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not entry.get("is_valid"):
                continue
            puzzle_id = entry.get("id")
            score = entry.get("difficulty_score")
            if puzzle_id is None or score is None:
                continue
            rec = solved.setdefault(puzzle_id, {"score": score, "reps": set()})
            rec["reps"].add((board_type, prompt_type))
    return solved


def top_difficult_puzzles(solved: Dict[str, Dict[str, object]], top_n: int = 5) -> List[Tuple[str, float, Set[Tuple[str, str]]]]:
    ranked = sorted(
        ((pid, data["score"], data["reps"]) for pid, data in solved.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_n]


def main():
    target_dir = MODEL_RESULTS_DIR
    if not os.path.exists(target_dir):
        print(f"Results directory not found: {target_dir}")
        return

    solved = collect_solved_puzzles(target_dir)
    if not solved:
        print("No solved puzzles found in evaluations.")
        return

    hardest = top_difficult_puzzles(solved, top_n=5)
    print("Top 5 most difficult solved puzzles (by difficulty_score):")
    for rank, (puzzle_id, score, reps) in enumerate(hardest, start=1):
        reps_str = ", ".join(sorted(f"{b} (prompt: {p})" for b, p in reps))
        print(f"{rank}. {puzzle_id} - difficulty {score:.3f} - solved via: {reps_str}")


if __name__ == "__main__":
    main()
