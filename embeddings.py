import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

w = np.load('model.npz')
emb = torch.tensor(w['embedding_weight'], dtype = torch.float32, device = device)


def emb_look(IDs):
    return emb[IDs]

def positinal_encoding(sqn_len, d_model):
    pos = torch.arange(sqn_len, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device)
    div_term = torch.exp(i * -(torch.log(torch.tensor(10000.0, device=device)) / d_model))
    
    pe = torch.zeros((sqn_len, d_model), device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe