
import json
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import load_data_random, load_data_fs

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, ImageDataset_FS, my_dataloader
from Features import Features
from ResNet import get_model
from TrainModel import train_model

def metric_acc_batch(y_hat, y):
    y_hat = y_hat.cpu().data.numpy()
    y_hat = 1 * (y_hat >= 0)
    y = y.cpu().data.numpy()

    out = np.zeros((2))
    out[0] = np.sum(y == y_hat)
    out[1] = len(y)
    
    return out

def metric_acc_agg(counts_list = None):
    if counts_list is None:
        return ['Acc']
    else:
        correct = 0
        total = 0
        
        for counts in counts_list:
            correct += counts[0]
            total += counts[1]
            
        return [correct / total]
        
def train(mode, label1, label2, spurious, trial,
            mp_override = None, lr_override = None, bs_override = None,
            model_dir = None):

    # Setup the output directory
    if model_dir is None:
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(label1, label2, spurious, mode, trial)
    os.system('rm -rf {}'.format(model_dir))
    Path(model_dir).mkdir(parents = True, exist_ok = True)

    name = '{}/model'.format(model_dir)
        
    # Load the chosen images for this pair
    data_dir = '{}/{}-{}/{}/train'.format(get_data_dir(), label1, label2, spurious )
    with open('{}/splits.json'.format(data_dir), 'rb') as f:
        splits = json.load(f)
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
    
    ids = []
    for key in splits:
        for id in splits[key]:
            ids.append(id)
        
    # Get the ids of the training images for this experiment
    # By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    ids_train, ids_val = train_test_split(ids, test_size = 0.1)
    
    # Load defaults
    if 'transfer' in mode.split('-'):
        lr = 0.001
    elif 'tune' in mode.split('-'):
        lr = 0.0001
    else:
        lr = None
    
    mode_param = 0.0
    batch_size = 64
    feature_hook = None

    # Load the the data specified by mode for each Image ID
    if mode in ['initial-transfer', 'initial-tune']:
        names = {}
        names['1s'] = {'orig': 1.0}
        names['1ns'] = {'orig': 1.0}
        names['0s'] = {'orig': 1.0}
        names['0ns'] = {'orig': 1.0}
    elif mode in ['removed-tune']:
        names = {}
        names['1s'] = {'orig': 1.0, 'spurious-box': 1.0}
        names['1ns'] = {'orig': 1.0}
        names['0s'] = {'orig': 1.0, 'spurious-box': 1.0}
        names['0ns'] = {'orig': 1.0}
    elif mode in ['added-tune']:
        names = {}
        names['1s'] = {'orig': 1.0}
        names['1ns'] = {'orig': 1.0, '1ns+1s': 1.0}
        names['0s'] = {'orig': 1.0}
        names['0ns'] = {'orig': 1.0, '0ns+1s': 1.0}
    elif mode in ['combined-tune']:
        names = {}
        names['1s'] = {'orig': 1.0, 'spurious-box': 1.0}
        names['1ns'] = {'orig': 1.0, '1ns+1s': 1.0}
        names['0s'] = {'orig': 1.0, 'spurious-box': 1.0}
        names['0ns'] = {'orig': 1.0, '0ns+1s': 1.0}
    elif mode in ['fs-tune']:
        pass
    else:
        print('Error: Unrecognized mode')
        sys.exit(0)
        
    # Apply over-rides
    if mp_override is not None:
        mode_param = mp_override
        
    if lr_override is not None:
        lr = lr_override
        
    if bs_override is not None:
        batch_size = bs_override
    
    # Setup the data loaders
    if mode in []:
        pass # Used for methods that pair the real and counterfactual examples
    elif mode in ['fs-tune']:
        files_train, labels_train, contexts_train = load_data_fs(ids_train, images, splits)
        files_val, labels_val, contexts_val = load_data_fs(ids_val, images, splits)

        datasets = {}
        datasets['train'] = ImageDataset_FS(files_train, labels_train, contexts_train)
        datasets['val'] = ImageDataset_FS(files_val, labels_val, contexts_val)
    else:
        files_train, labels_train = load_data_random(ids_train, images, splits, names)
        files_val, labels_val = load_data_random(ids_val, images, splits, names)

        datasets = {}
        datasets['train'] = ImageDataset(files_train, labels_train)
        datasets['val'] = ImageDataset(files_val, labels_val)

    dataloaders = {}
    dataloaders['train'] = my_dataloader(datasets['train'], batch_size = batch_size)
    dataloaders['val'] = my_dataloader(datasets['val'], batch_size = batch_size)
    
    # Setup the model and optimization process
    parent_transfer = './Models/{}-{}/{}/initial-transfer/trial{}/model.pt'.format(label1, label2, spurious, trial)
    parent_tune = './Models/{}-{}/{}/initial-tune/trial{}/model.pt'.format(label1, label2, spurious, trial)
    if mode == 'initial-transfer':
        model, optim_params = get_model(mode = 'transfer', parent = 'pretrained')
    elif mode in ['initial-tune', 'removed-tune', 'added-tune', 'combined-tune']:
        model, optim_params = get_model(mode = 'tune', parent = parent_transfer)
    elif mode in ['fs-tune']:
        model, optim_params = get_model(mode = 'tune', parent = parent_tune)
    else:
        print('Train.py: Could not determine trainable parameters')
        sys.exit(0)
    
    # Setup the feature hook for getting the representations
    if mode in ['fs-tune']:
        feature_hook = Features()
        handle = list(model.modules())[66].register_forward_hook(feature_hook) # Warning:  this is specific to ResNet18

    model.cuda()

    if mode in ['fs-tune']:
        metric_loss = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    else:
        metric_loss = torch.nn.BCEWithLogitsLoss()
    
    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name,
                        lr_init = lr, select_cutoff = 5, decay_max = 1,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    os.system('rm -rf {}'.format(name)) # Clean up the model history saved during training

if __name__ == '__main__':
    
    mode = sys.argv[1]
    label1 = sys.argv[2]
    label2 = sys.argv[3]
    spurious = sys.argv[4]
    trials = sys.argv[5].split(',')
    
    for trial in trials:
        train(mode, label1, label2, spurious, trial)
