import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from model import build_llama3_model
from data_preprocessing import preprocess_data
from tokenizer import create_tokenizer

def pretrain(model_forward, dataset, num_epochs=1, batch_size=2, learning_rate=1e-5):

  dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
  optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
      optimizer.zero_grad()
      input_ids = batch["input_ids"].to(torch.long)
      logits = model_forward(input_ids[:, :-1])

      loss = criterion(logits, input_ids[:, 1:])
      loss.backward()
      optimizer.step()
      if batch_idx % 10 ==0:
          print(f"Epoch: {epoch+1}, Batch: {batch_idx}, loss: {loss.item()}")

if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    with open("tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)

    tokenizer = create_tokenizer(config["tokenizer_path"], tokenizer_config["special_tokens"] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)])
    processed_dataset = preprocess_data(tokenizer)
    model = torch.load(config["model_path"])
    model_forward = build_llama3_model(config, model)

    pretrain(model_forward, processed_dataset)