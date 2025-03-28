import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json
from pathlib import Path

def create_tokenizer(tokenizer_path, special_tokens):
    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    tokenizer = tiktoken.Encoding(
        name=Path(tokenizer_path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
    )
    return tokenizer


if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    with open("tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)

    tokenizer = create_tokenizer(config["tokenizer_path"], tokenizer_config["special_tokens"] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)])


    text = "hello world!"
    tokens = tokenizer.encode(text)
    print("Encoded tokens:", tokens)

    decoded_text = tokenizer.decode(tokens)
    print("Decoded text:", decoded_text)