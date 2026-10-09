import torch
import numpy as np
import torch.nn.functional as F
import math
from transformer import transformer_forward
from tokenizer import encode, decode, char

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

d_dim = 384
d_head = 6
squ_len = 128
vocab_size = len(char)

def positinal_encoding(sqn_len, d_model):
    pos = torch.arange(sqn_len, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device)
    div_term = torch.exp(i * -(torch.log(torch.tensor(10000.0, device=device)) / d_model))
    pe = torch.zeros((sqn_len, d_model), device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

def load_layer_weights(w, layer_prefix):
    return {
        'wq': torch.tensor(w[f'{layer_prefix}_WQ'], dtype=torch.float32, device=device),
        'bq': torch.tensor(w[f'{layer_prefix}_bQ'], dtype=torch.float32, device=device),
        'wk': torch.tensor(w[f'{layer_prefix}_WK'], dtype=torch.float32, device=device),
        'bk': torch.tensor(w[f'{layer_prefix}_bK'], dtype=torch.float32, device=device),
        'wv': torch.tensor(w[f'{layer_prefix}_WV'], dtype=torch.float32, device=device),
        'bv': torch.tensor(w[f'{layer_prefix}_bV'], dtype=torch.float32, device=device),
        'wo': torch.tensor(w[f'{layer_prefix}_WO'], dtype=torch.float32, device=device),
        'bo': torch.tensor(w[f'{layer_prefix}_bO'], dtype=torch.float32, device=device),
        'gamma1': torch.tensor(w[f'{layer_prefix}_gamma1'], dtype=torch.float32, device=device),
        'beta1': torch.tensor(w[f'{layer_prefix}_beta1'], dtype=torch.float32, device=device),
        'w1': torch.tensor(w[f'{layer_prefix}_W1'], dtype=torch.float32, device=device),
        'b1': torch.tensor(w[f'{layer_prefix}_b1'], dtype=torch.float32, device=device),
        'w2': torch.tensor(w[f'{layer_prefix}_W2'], dtype=torch.float32, device=device),
        'b2': torch.tensor(w[f'{layer_prefix}_b2'], dtype=torch.float32, device=device),
        'gamma2': torch.tensor(w[f'{layer_prefix}_gamma2'], dtype=torch.float32, device=device),
        'beta2': torch.tensor(w[f'{layer_prefix}_beta2'], dtype=torch.float32, device=device),
    }

@torch.no_grad()
def generate(prompt, num_tokens, checkpoint_path, temperature=0.8, top_k=20):
    w = np.load(checkpoint_path)

    WO = torch.tensor(w['W_out'], dtype=torch.float32, device=device)
    BO = torch.tensor(w['b_out'], dtype=torch.float32, device=device)
    emb = torch.tensor(w['embedding_weight'], dtype=torch.float32, device=device)

    weights = [load_layer_weights(w, f'b{i}') for i in range(1, 7)]

    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)

    for _ in range(num_tokens):
        context = tokens[-squ_len:]
        T = context.shape[0]

        x_emb = emb[context].unsqueeze(0)          # [1, T, d_dim]
        PE = positinal_encoding(T, d_dim)
        input_seq = (x_emb * (d_dim ** 0.5)) + PE

        out = input_seq
        for wts in weights:
            out, _ = transformer_forward(out, wts, d_dim, d_head)

        pred = out @ WO + BO                        # [1, T, vocab_size]
        logits = pred[0, -1, :] / temperature

        if top_k is not None:
            v, idx = torch.topk(logits, top_k)
            probs = F.softmax(v, dim=-1)
            next_token = idx[torch.multinomial(probs, 1)]
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        tokens = torch.cat([tokens, next_token.view(1)])

    out_ids = tokens.tolist()
    return ''.join(decode(out_ids))


if __name__ == "__main__":
    checkpoint_path = 'model_step10000_09-25-09.npz'
    prompt = input("Prompt: ")
    num_tokens = int(input("How many characters to generate: "))
    temperature = float(input("Temperature : ") or 0.8)

    result = generate(prompt, num_tokens, checkpoint_path, temperature=temperature, top_k=20)
    print("\n--- Generated ---\n")
    print(result)