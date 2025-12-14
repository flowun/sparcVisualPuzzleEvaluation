import base64
from sparc.prompt import generate_prompt

from prompts.prompts import get_prompt

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def decode_base64_to_image(b64_string, output_path):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_string))

def create_payload_with_image(prompt_type, board_type, image_path, data, model, temperature, max_tokens=10000, top_p=0.95, top_k=20, seed=42):
    if image_path is None:  # default to standard sparc evaluation if no board image and prompt are provided
        return create_textual_payload(data, model, temperature, max_tokens, top_p, top_k, seed)

    text_prompt = get_prompt(prompt_type, board_type, data)
    b64_image = encode_image_to_base64(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    return payload

def create_textual_payload(data, model, temperature, max_tokens, top_p, top_k, seed):
    # uses the standard SPARC textual prompt generation (alternative (improved) textual prompt from the paper: https://arxiv.org/pdf/2505.16686)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": generate_prompt(data)
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    return payload
