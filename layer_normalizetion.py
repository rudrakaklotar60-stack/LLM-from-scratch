import torch

def layer_norm_forward(x, gamma, beta, eps):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, correction=0) 
    x_hat = (x - mean) / torch.sqrt(var + eps)
    output = x_hat * gamma + beta 
    cache = (x, x_hat, mean, var, gamma, eps)
    return output, cache    

def layer_norm_back(dx, cache, d_model):
    x, x_hat, mean, var, gamma, eps = cache
    std = 1 / torch.sqrt(var + eps)
    
    dbeta = torch.sum(dx, dim=(0, 1))
    dgamma = torch.sum(dx * x_hat, dim=(0, 1))
    
    term1 = (-1/d_model) * (var + eps)**(-1.5) * (x - mean) * torch.sum(dx * gamma * (x - mean), dim=-1, keepdim=True)
    term2 = (-1/d_model) * std * torch.sum(dx * gamma, dim=-1, keepdim=True)
    term3 = dx * gamma * std
    
    d_x = term1 + term2 + term3
    
    return d_x, dgamma, dbeta