
import torch

def gs_loss(rep, rep_prime, y, y_prime, prob):

    n = rep.shape[0]
    
    d_rep = rep_prime - rep
    d_rep = torch.reshape(d_rep, (n, -1))
    
    d_y = y_prime - y
    d_y = torch.reshape(d_y, (n, -1))

    grad_data = d_y * d_rep
    grad_data = torch.reshape(grad_data, (n, -1))
    grad_data = torch.nn.functional.normalize(grad_data)
            
    grad_model = torch.autograd.grad(prob, rep, grad_outputs = torch.ones(prob.shape).to('cuda'), create_graph = True)[0]
    grad_model = torch.reshape(grad_model, (n, -1))
    grad_model = torch.nn.functional.normalize(grad_model)
        
    cs = torch.sum(grad_data * grad_model, dim = 1)
    
    weights = torch.sum(torch.square(grad_data), dim = 1) #Not all images have a counterfactual version, this ignores those images
    loss = torch.sum(weights * 0.5 * (1 - cs)) / torch.sum(weights).clamp_min(1) #It is possible the at draw an entire minibatch with no counterfactuals, this prevents the loss from breaking
    return loss
