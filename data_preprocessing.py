from datasets import load_dataset
import torch
import json
from tokenizer import create_tokenizer

def preprocess_data(tokenizer, max_length=1024):
    ds = load_dataset("rojagtap/bookcorpus")

    def tokenize_function(examples):
      tokens = [128000] + tokenizer.encode(examples['text'])[:max_length-1]
      return {"input_ids": tokens}

    tokenized_dataset = ds.map(tokenize_function, batched=False)

    def pad_function(examples):
        padded_tokens = [torch.tensor(tokens).long() for tokens in examples['input_ids']]
        padded_tokens = torch.nn.utils.rnn.pad_sequence(padded_tokens, batch_first=True)
        return {'input_ids': padded_tokens}

    padded_dataset = tokenized_dataset.map(pad_function, batched=True)

    return padded_dataset



if __name__ == '__main__':
   with open("config.json", "r") as f:
        config = json.load(f)
   with open("tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)
   tokenizer = create_tokenizer(config["tokenizer_path"], tokenizer_config["special_tokens"] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)])

   processed_dataset = preprocess_data(tokenizer)

   print(processed_dataset['train'][0])
   print(f"Dataset shape {processed_dataset['train'].shape}")