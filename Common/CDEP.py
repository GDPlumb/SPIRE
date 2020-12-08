
# Modified Version of:  https://github.com/csinva/hierarchical-dnn-interpretations/tree/master/acd/scores

import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.special import expit as sigmoid

def cdep_loss(x, x_prime, model):
    model.eval()
    mask = torch.max(1.0 * (x != x_prime), 1, keepdim = True, out = None)[0]
    changed = mask * x
    same = (1 - mask) * x
    score_changed, score_same = cd_propagate_resnet(changed, same, model)
    loss = torch.nn.functional.softmax(torch.stack((score_changed.view(-1), score_same.view(-1)), dim =1), dim = 1)[:,0].mean()
    #loss = torch.mean(torch.abs(score_schanged))
    return loss

def cd_propagate_resnet(rel, irrel, model):
    '''Propagate a resnet architecture
    each BasicBlock passes its input through to its output (might need to downsample)
    note: the bigger resnets use BottleNeck instead of BasicBlock
    '''
    mods = list(model.modules())
    '''
    # mods[1:5]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    # mods[5, 18, 34, 50]
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    '''
        
    rel, irrel = cd_generic(mods[1:5], rel, irrel)

    lay_nums = [5, 18, 34, 50]
    for lay_num in lay_nums:
        for basic_block in mods[lay_num]:
            rel, irrel = propagate_basic_block(rel, irrel, basic_block)
    
    # final things after BasicBlocks
    rel, irrel = cd_generic(mods[-2:], rel, irrel)
    return rel, irrel

def cd_generic(mods, relevant, irrelevant):
    '''Helper function for cd which loops over modules and propagates them
    based on the layer name
    '''
    for i, mod in enumerate(mods):
        t = str(type(mod))
        if 'Conv2d' in t:
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'Linear' in t:
            relevant = relevant.reshape(relevant.shape[0], -1)
            irrelevant = irrelevant.reshape(irrelevant.shape[0], -1)
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'ReLU' in t:
            relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
        elif 'AvgPool' in t or 'NormLayer' in t or 'Dropout' in t \
             or 'ReshapeLayer' in t or ('modularize' in t and 'Transform' in t): # custom layers
            relevant, irrelevant = propagate_independent(relevant, irrelevant, mod)
        elif 'Pool' in t and not 'AvgPool' in t:
            relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod)
        elif 'BatchNorm2d' in t:
            relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mod)
    return relevant, irrelevant

def propagate_conv_linear(relevant, irrelevant, module):
    '''Propagate convolutional or linear layer
    Apply linear part to both pieces
    Split bias based on the ratio of the absolute sums
    '''
    device = relevant.device
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel) + 1e-20 # add a small constant so we don't divide by 0
    prop_irrel = torch.abs(irrel) + 1e-20 # add a small constant so we don't divide by 0
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


def propagate_batchnorm2d(relevant, irrelevant, module):
    '''Propagate batchnorm2d operation
    '''
    device = relevant.device
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias
    prop_rel = torch.abs(rel) + 1e-20 # add a small constant so we don't divide by 0
    prop_irrel = torch.abs(irrel) + 1e-20 # add a small constant so we don't divide by 0
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_rel[torch.isnan(prop_rel)] = 0
    rel = rel + torch.mul(prop_rel, bias)
    irrel = module(relevant + irrelevant) - rel
    return rel, irrel

def propagate_pooling(relevant, irrelevant, pooler):
    '''propagate pooling operation
    '''
    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)
    
    # unpooling function
    def unpool(tensor, indices):
        '''Unpool tensor given indices for pooling
        '''
        batch_size, in_channels, H, W = indices.shape
        output = torch.ones_like(indices, dtype=torch.float)
        for i in range(batch_size):
            for j in range(in_channels):
                output[i, j] = tensor[i, j].flatten()[indices[i, j].flatten()].reshape(H, W)
        return output
    
    rel, irrel = unpool(relevant, both_ind), unpool(irrelevant, both_ind)
    return rel, irrel

def propagate_independent(relevant, irrelevant, module):
    '''use for things which operate independently
    ex. avgpool, layer_norm, dropout
    '''
    return module(relevant), module(irrelevant)

def propagate_relu(relevant, irrelevant, activation):
    '''propagate ReLu nonlinearity
    '''
    device = relevant.device
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

def propagate_basic_block(rel, irrel, module):
    '''Propagate a BasicBlock (used in the ResNet architectures)
    This is what the forward pass of the basic block looks like
    identity = x

    out = self.conv1(x) # 1
    out = self.bn1(out) # 2
    out = self.relu(out) # 3
    out = self.conv2(out) # 4
    out = self.bn2(out) # 5

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    '''
    rel_identity, irrel_identity = rel.detach().clone(), irrel.detach().clone() #deepcopy(rel), deepcopy(irrel) #
    rel, irrel = cd_generic(list(module.modules())[1:6], rel, irrel)
    
    if module.downsample is not None:
        rel_identity, irrel_identity = cd_generic(module.downsample.modules(), rel_identity, irrel_identity)

    rel += rel_identity
    irrel += irrel_identity
    rel, irrel = propagate_relu(rel, irrel, module.relu)
    
    return rel, irrel
