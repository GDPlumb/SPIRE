
import sys
import torch
from torchvision.models import resnet18

from Features import Features

class LinearModel(torch.nn.Module):
    def __init__(self, W, b):
        super(LinearModel, self).__init__()
        linear = torch.nn.Linear(W.shape[0], W.shape[1], bias = True)
        linear.weight = torch.nn.Parameter(W)
        linear.bias = torch.nn.Parameter(b)
        self.linear = linear

    def forward(self, x):
        out = self.linear(x)
        return out

def get_model(out_features = 1, mode = 'tune', parent = 'pretrained'):
    # Load the model
    model = resnet18(pretrained = (parent == 'pretrained'))
    # Change the classification layer
    model.fc = torch.nn.Linear(in_features = 512, out_features = out_features)
    # Load the in the parent model weights
    if parent != 'pretrained':
        model.load_state_dict(torch.load(parent))
    # Setup the trainable parameters
    if mode == 'tune':
        return model, model.parameters()
    elif mode == 'transfer':
        for param in model.parameters():
            param.requires_grad = False
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model, model.fc.parameters()
    elif mode == 'eval':
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        print('ResNet.py: Could not determine trainable parameters')
        sys.exit(0)

def get_features(model):
    feature_hook = Features()
    handle = list(model.modules())[66].register_forward_hook(feature_hook)
    return feature_hook
    
def get_lm(model, label_indices = None):
    if label_indices is not None:
        lm = LinearModel(model.fc.weight[label_indices, :], model.fc.bias[label_indices])
    else:
        lm = LinearModel(model.fc.weight, model.fc.bias)
    return lm

def set_lm(model, lm, label_indices = None):
    with torch.no_grad():
        if label_indices is not None:
            model.fc.weight[label_indices, :] = lm.linear.weight
            model.fc.bias[label_indices] = lm.linear.bias
        else:
            model.fc.weight = lm.linear.weight
            model.fc.bias = lm.linear.bias
