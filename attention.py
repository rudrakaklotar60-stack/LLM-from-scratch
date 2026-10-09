import torch

def softMaxBack(grad, softout):
    dx = softout * (grad - torch.sum(grad * softout, dim=-1, keepdim=True))
    return dx
    
def Multi_Head_Attention_Forward(input, wq, wk, wv, wo, bq, bk, bv, bo, d_dim, d_head):
    B, T, C = input.shape
    hehe = d_dim // d_head
    
    out_Q = (input @ wq + bq).view(B, T, d_head, hehe).transpose(1, 2)
    out_K = (input @ wk + bk).view(B, T, d_head, hehe).transpose(1, 2)
    out_V = (input @ wv + bv).view(B, T, d_head, hehe).transpose(1, 2)
    
    scores = (out_Q @ out_K.transpose(-2, -1)) / (hehe ** 0.5)

    mask = torch.triu(torch.ones((T, T), device=input.device) * float('-inf'), diagonal=1)
    scores += mask

    max_vals = torch.max(scores, dim=-1, keepdim=True).values
    e_x = torch.exp(scores - max_vals)
    weights = e_x / torch.sum(e_x, dim=-1, keepdim=True)
    
    merged = weights @ out_V #  [B, Heads, T, hehe]
    
    merged = merged.transpose(1, 2).contiguous().view(B, T, C)
    
    output = merged @ wo + bo
    
    cache = (input, out_Q, out_K, out_V, wq, wk, wv, wo, merged, weights)
    return output, cache

def Multi_Head_Attention_Backward(grad, cache, d_dim, d_head):
    input, out_Q, out_K, out_V, wq, wk, wv, wo, merged, weights = cache
    B, T, C = input.shape
    hehe = d_dim // d_head
    
    dbO = torch.sum(grad, dim=(0, 1))
    dwO = merged.view(-1, C).T @ grad.view(-1, C)
    
    d_merged = grad @ wo.T 
    d_merged = d_merged.view(B, T, d_head, hehe).transpose(1, 2) # [B, Heads, T, hehe]
    
    d_weights = d_merged @ out_V.transpose(-2, -1)
    d_out_V = weights.transpose(-2, -1) @ d_merged
    
    d_scores = softMaxBack(d_weights, weights)
    
    d_out_Q = (d_scores @ out_K) / (hehe ** 0.5)
    d_out_K = (d_scores.transpose(-2, -1) @ out_Q) / (hehe ** 0.5)
    
    d_out_Q = d_out_Q.transpose(1, 2).contiguous().view(B, T, C)
    d_out_K = d_out_K.transpose(1, 2).contiguous().view(B, T, C)
    d_out_V = d_out_V.transpose(1, 2).contiguous().view(B, T, C)
    
    dwQ = input.view(-1, C).T @ d_out_Q.view(-1, C)
    dwK = input.view(-1, C).T @ d_out_K.view(-1, C)
    dwV = input.view(-1, C).T @ d_out_V.view(-1, C)
    
    dbQ = torch.sum(d_out_Q, dim=(0, 1))
    dbK = torch.sum(d_out_K, dim=(0, 1))
    dbV = torch.sum(d_out_V, dim=(0, 1))
    
    dx = d_out_Q @ wq.T + d_out_K @ wk.T + d_out_V @ wv.T
    
    return dx, dwQ, dwK, dwV, dwO, dbQ, dbK, dbV, dbO