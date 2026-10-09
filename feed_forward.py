import torch

def gelu_forward(x):
    return (0.5) * x * (1 + torch.tanh(((2 / torch.pi) ** 0.5) * (x + 0.044715 * x**3)))
    
def gelu_backward(x):
    return (0.5) * (1 + torch.tanh(0.0356774 * x**3 + 0.797885 * x)) + (0.0535161 * x**3 + 0.398942 * x) * (1 - (torch.tanh(0.0356774 * x**3 + 0.797885 * x)**2))

def feed_forward_forward(input, w1, b1, w2, b2):
    out1 = input @ w1 + b1 
    out1_gelu = gelu_forward(out1)
    out2 = out1_gelu @ w2 + b2
    cache = (input, out1, out1_gelu, out2, w1, w2)
    return out2, cache
    
def feed_forward_backward(grad, cache):
    input, out1, out1_gelu, out2, w1, w2 = cache
    
    d_out1_gelu = grad @ w2.T
    
    dw2 = out1_gelu.view(-1, out1_gelu.shape[-1]).T @ grad.view(-1, grad.shape[-1])
    db2 = torch.sum(grad, dim=(0, 1))
    
    d_out1 = d_out1_gelu * gelu_backward(out1)
    
    d_input = d_out1 @ w1.T
    dw1 = input.view(-1, input.shape[-1]).T @ d_out1.view(-1, d_out1.shape[-1])
    db1 = torch.sum(d_out1, dim=(0, 1))
   
    return d_input, dw1, db1, dw2, db2