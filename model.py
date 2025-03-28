import torch
import json

def rms_norm(tensor, norm_weights, norm_eps=1e-05):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights


def build_llama3_model(config, model):
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    rope_theta = torch.tensor(config["rope_theta"])

    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])

    def forward(tokens):
       token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
       final_embedding = token_embeddings_unnormalized

       freqs = 1.0 / (rope_theta ** (torch.tensor(range(64))/64))
       freqs_for_each_token = torch.outer(torch.arange(tokens.shape[1]), freqs)
       freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
       for layer in range(n_layers):
           qkv_attention_store = []
           layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
           q_layer = model[f"layers.{layer}.attention.wq.weight"]
           q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
           k_layer = model[f"layers.{layer}.attention.wk.weight"]
           k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
           v_layer = model[f"layers.{layer}.attention.wv.weight"]
           v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
           w_layer = model[f"layers.{layer}.attention.wo.weight"]
           for head in range(n_heads):
               q_layer_head = q_layer[head]
               k_layer_head = k_layer[head//4]
               v_layer_head = v_layer[head//4]
               q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
               k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
               v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
               q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
               q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
               q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:tokens.shape[1]])
               q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
               k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
               k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
               k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:tokens.shape[1]])
               k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
               qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
               mask = torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"))
               mask = torch.triu(mask, diagonal=1)
               qk_per_token_after_masking = qk_per_token + mask
               qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
               qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
               qkv_attention_store.append(qkv_attention)


           stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
           w_layer = model[f"layers.{layer}.attention.wo.weight"]
           embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
           embedding_after_edit = final_embedding + embedding_delta
           embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
           w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
           w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
           w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
           output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
           final_embedding = embedding_after_edit+output_after_feedforward

       final_embedding = rms_norm(final_embedding, model["norm.weight"])
       logits = torch.matmul(final_embedding[:, -1, :], model["output.weight"].T)
       return logits

    return forward

if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    model = torch.load(config["model_path"])
    model_forward = build_llama3_model(config, model)

    tokens = torch.tensor([[128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]])
    logits = model_forward(tokens)
    next_token = torch.argmax(logits, dim=-1)
    print(f"Predicted token: {next_token.item()}")