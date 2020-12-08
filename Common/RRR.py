
import torch

def rrr_loss(x, x_prime, prob):
    mask = torch.max(1.0 * (x != x_prime), 1, keepdim = True, out = None)[0]
    grad = torch.autograd.grad(prob, x, grad_outputs = torch.ones(prob.shape).to('cuda'), create_graph = True)[0]
    loss = torch.norm(mask * grad, 2)
    return loss
