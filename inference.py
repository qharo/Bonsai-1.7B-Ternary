#!/usr/bin/env python3
"""Simple inference with the local Ternary-Bonsai-1.7B unpacked weights + transformers."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "models/Ternary-Bonsai-1.7B-unpacked"
    prompt = "What's the weather like, usually, in Nice, France?'"

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")
    print(f"\nPrompt: {prompt}\n")

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids.shape[1]}")

    print("\nGenerating...\n")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"\n> {prompt}")
    print(response)


if __name__ == "__main__":
    main()
