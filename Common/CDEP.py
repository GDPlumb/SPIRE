
# Modified from original source:  https://github.com/laura-rieger/deep-explanation-penalization/blob/master/src/cd.py
from copy import deepcopy
import torch

stabilizing_constant = 10e-20

def propagate_conv_linear(relevant, irrelevant, module, device='cuda'):
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel + stabilizing_constant
    
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)

def propagate_AdaptiveAvgPool2d(relevant, irrelevant, module,  device='cuda'):
    rel = module(relevant)
    irrel = module(irrelevant)
    return rel, irrel

def propagate_relu(relevant, irrelevant, activation, device='cuda'):
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    zeros = torch.zeros(relevant.size()).to(device)
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score

def propagate_pooling(relevant, irrelevant, pooler, model_type='mnist'):
    if  model_type == 'vgg':
        unpool = torch.nn.MaxUnpool2d(kernel_size=pooler.kernel_size, stride=pooler.stride)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=(pooler.kernel_size, pooler.kernel_size), stride=(pooler.stride, pooler.stride), count_include_pad=False)
        window_size = 4

    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)
    ones_out = torch.ones_like(both)
    size1 = relevant.size()
    mask_both = unpool(ones_out, both_ind, output_size=size1)

    # relevant
    rel = mask_both * relevant
    rel = avg_pooler(rel) * window_size

    # irrelevant
    irrel = mask_both * irrelevant
    irrel = avg_pooler(irrel) * window_size
    return rel, irrel

def propagate_dropout(relevant, irrelevant, dropout):
    return dropout(relevant), dropout(irrelevant)

# get contextual decomposition scores for the masked/chosen part of the image
def cd(mask, im, model, model_type='vgg'):
 
    # Setup the Model
    model.eval()
    
    # Split the image using the mask
    relevant = mask * im
    irrelevant = (1 - mask) * im
    
    mods = list(model.modules())
    for i, mod in enumerate(mods):
        t = str(type(mod))
        if 'Conv2d' in t or 'Linear' in t:
            if 'Linear' in t:
                relevant = relevant.view(relevant.size(0), -1)
                irrelevant = irrelevant.view(irrelevant.size(0), -1)
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'ReLU' in t:
            relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
        elif 'MaxPool2d' in t:
            relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod, model_type = model_type)
        elif 'Dropout' in t:
            relevant, irrelevant = propagate_dropout(relevant, irrelevant, mod)
        elif 'AdaptiveAvgPool2d' in t:
            relevant, irrelevant = propagate_AdaptiveAvgPool2d(relevant, irrelevant, mod)
            
    return relevant, irrelevant

def cdep_loss(x, x_prime, model):
    mask = torch.max(1.0 * (x != x_prime), 1, keepdim = True, out = None)[0]
    
    score_changed, score_same = cd(mask, x, model)
    
    loss = torch.nn.functional.softmax(torch.stack((score_changed.view(-1), score_same.view(-1)), dim =1), dim = 1)[:,0].mean()
    #loss = torch.mean(torch.abs(score_changed))
    
    return loss
