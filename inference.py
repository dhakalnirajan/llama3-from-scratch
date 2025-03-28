import torch
import json
from model import build_llama3_model
from tokenizer import create_tokenizer

def infer(model_forward, tokenizer, prompt, max_new_tokens=20):
    tokens = [128000] + tokenizer.encode(prompt)
    tokens = torch.tensor([tokens])

    generated_tokens = []

    for _ in range(max_new_tokens):
        logits = model_forward(tokens)
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens.append(next_token.item())
        tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return tokenizer.decode(generated_tokens)


if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    with open("tokenizer_config.json", "r") as f:
      tokenizer_config = json.load(f)
    tokenizer = create_tokenizer(config["tokenizer_path"], tokenizer_config["special_tokens"] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)])
    model = torch.load(config["model_path"])
    model_forward = build_llama3_model(config, model)

    prompt = "the answer to the ultimate question of life, the universe, and everything is "
    output = infer(model_forward, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")