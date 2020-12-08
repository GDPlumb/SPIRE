
import torch

def gs_loss(rep, rep_prime, prob):

    n = rep.shape[0]
    
    grad_data = rep - rep_prime
    grad_data = torch.reshape(grad_data, (n, -1))
    grad_data = torch.nn.functional.normalize(grad_data)
            
    grad_model = torch.autograd.grad(prob, rep, grad_outputs = torch.ones(prob.shape).to('cuda'), create_graph = True)[0]
    grad_model = torch.reshape(grad_model, (n, -1))
    grad_model = torch.nn.functional.normalize(grad_model)
        
    cs = torch.sum(grad_data * grad_model, dim = 1)
    
    weights = torch.sum(torch.square(grad_data), dim = 1) #Not all images have a counterfactual version, this ignores those images
    loss = torch.sum(weights * 0.5 * (1 - cs)) / torch.sum(weights).clamp_min(1) #It is possible the at draw an entire minibatch with no counterfactuals, this prevents the loss from breaking
    return loss
