import base64
import requests

from prompts.prompts import get_prompt

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def decode_base64_to_image(b64_string, output_path):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_string))

def create_payload_with_image(prompt_type, board_type, image_path, data, model, temperature, max_tokens=10000, top_p=0.95, top_k=20, seed=42):
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

"""
# doesn't work with well with qwen3-vl-thinking, better increase max tokens
def modify_payload_to_force_final_answer(payload, results):
    # print("old payload:", payload)
    # print("output:", results["choices"][0]["message"]["content"])
    # if model didn't finish thinking and providing the answer in the max tokens, we force it to finish fast
    answer_forcing_prompt = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"
    import json
    # print("old payload:", json.dumps(payload, indent=2))
    payload["max_tokens"] += 500
    payload["messages"].append({"role": "assistant", "content": results["choices"][0]["message"]["content"] + answer_forcing_prompt})
    # visualize payload in console with indentation
    # print("new payload:", json.dumps(payload, indent=2))
    return payload
"""
"""
def modify_payload_to_force_final_answer(payload, results):
    answer_forcing_prompt = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"
    new_payload = {
        "model": payload["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": payload['messages'][0]['content'][0]['image_url']['url']
                        }
                    },
                    {
                        "type": "text",
                        "text": payload['messages'][0]['content'][1]['text']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": results["choices"][0]["message"]["content"] + answer_forcing_prompt}
        ],
        "max_tokens": payload["max_tokens"] + 500,
        "temperature": payload["temperature"],
        "top_p": payload["top_p"],
        "top_k": payload["top_k"],
    }
    return new_payload
"""
