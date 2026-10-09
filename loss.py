import torch

def cross_entropy(pred, y_true):
    pred = pred - torch.max(pred, dim=-1, keepdim=True).values
    expo = torch.exp(pred)
    prob = expo / torch.sum(expo, dim=-1, keepdim=True)
    
    total_elements = pred.shape[0] * pred.shape[1]
    loss = -torch.sum((y_true * torch.log(prob + 1e-9))) / total_elements
    cache = (prob, y_true, total_elements)
    return loss, cache
    
def loss_backward(cache):
    prob, y_true, total_elements = cache
    logits = (prob - y_true) / total_elements
    return logits