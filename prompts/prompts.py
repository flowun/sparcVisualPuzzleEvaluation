import re

def get_prompt(prompt_type, board_type, data=None):
    if data is not None:
        # correct a mistake in text dataset
        text_visualization = data.get("text_visualization", "")
        def _scale_dim(m):
            return f"{m.group(1)}{2 * int(m.group(2)) + 1}"
        text_visualization = re.sub(r"(width:\s*)(\d+)", _scale_dim, text_visualization)
        text_visualization = re.sub(r"(height:\s*)(\d+)", _scale_dim, text_visualization)
        data["text_visualization"] = text_visualization

    if prompt_type == "default_tr":  # Visual prompt from SPaRC paper (also gives textual coordinates)
        from .default import get_prompt as default_get_prompt
        return default_get_prompt(data, textual_representation=True)
    elif prompt_type == "default_no_tr":  # Visual prompt from SPaRC paper but without textual coordinates
        from .default import get_prompt as default_get_prompt
        return default_get_prompt(data, textual_representation=False)
    elif prompt_type == "prompt_engineering":  # Improved prompt & Vision only (no textual coordinates)
        from .prompt_engineering import get_prompt as prompt_engineering_get_prompt
        return prompt_engineering_get_prompt(data, board_type)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


if __name__ == "__main__":
    # Example usage
    from datasets import load_dataset
    dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")
    prompt = get_prompt("default_tr", "default", dataset[0])
    print(prompt)