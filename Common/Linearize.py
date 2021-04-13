
import numpy as np
import torch
from torch.utils.data import TensorDataset

from Dataset import my_dataloader
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
    
def get_lm(model, label_indices = None):
    # WARNING: this is specific to Resnet18
    if label_indices is not None:
        lm = LinearModel(model.fc.weight[label_indices, :], model.fc.bias[label_indices])
    else:
        lm = LinearModel(model.fc.weight, model.fc.bias)
    return lm
    
def get_data(model, dataloaders):
    
    feature_hook = Features()
    # WARNING: this is specific to Resnet18
    handle = list(model.modules())[66].register_forward_hook(feature_hook) 
    
    model.eval()
    model.cuda()
    
    # Get the representations for each of the images for this model
    data = {}
    labels = {}
    for phase in ['train', 'val']:
        data_tmp = []
        labels_tmp = []
        
        for batch in dataloaders[phase]:
            x = batch[0].cuda()
            y = batch[1]
            
            y_hat = model(x)
            rep = feature_hook.features

            data_tmp.append(rep.cpu().data.numpy())
            labels_tmp.append(y.data.numpy())
    
        data_tmp = np.squeeze(np.concatenate(data_tmp))
        labels_tmp = np.concatenate(labels_tmp)
        
        data[phase] = data_tmp
        labels[phase] = labels_tmp
        
    return data, labels

def get_loaders(data, labels, batch_size, label_indices = None):
    dataloaders = {}
    for phase in ['train', 'val']:
        data_tmp = torch.Tensor(data[phase])
        if label_indices is not None:
            labels_tmp = torch.Tensor(labels[phase][:, label_indices])
        else:
            labels_tmp = torch.Tensor(labels[phase])
        dataset = TensorDataset(data_tmp, labels_tmp)
        dataloaders[phase] = my_dataloader(dataset, batch_size = batch_size)
    return dataloaders
