
import sys
import torch
from torchvision.models import resnet18

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
