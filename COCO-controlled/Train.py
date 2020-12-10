
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import load_data_random, load_data_paired

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, ImageDataset_Paired, my_dataloader
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
        
def train(mode, main, spurious, p_correct, trial, p_main = 0.5, p_spurious = 0.5, n = 2000,
            mp_override = None, lr_override = None, bs_override = None,
            model_dir = None):

    # Setup the output directory
    if model_dir is None:
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
    os.system('rm -rf {}'.format(model_dir))
    Path(model_dir).mkdir(parents = True, exist_ok = True)

    name = '{}/model'.format(model_dir)
    
    # Load the chosen images for this pair
    data_dir = '{}/{}-{}/train'.format(get_data_dir(), main, spurious)
    with open('{}/splits.p'.format(data_dir), 'rb') as f:
        splits = pickle.load(f)
    both = splits['both']
    just_main = splits['just_main']
    just_spurious = splits['just_spurious']
    neither = splits['neither']
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
    
    # Find the number of images to get from each split
    num_main = int(n * p_main)
    num_spurious = int(n * p_spurious)

    num_both = int(p_correct * num_spurious)
    num_just_main = num_main - num_both
    num_just_spurious = num_spurious - num_both
    num_neither = n - num_both - num_just_main - num_just_spurious

    if num_both < 0 or num_just_main < 0 or num_just_spurious < 0 or num_neither < 0:
        print('Error: Bad Distribution Setup')
        print(num_both, num_just_main, num_just_spurious, num_neither)
        sys.exit(0)

    both_final = both[:num_both]
    just_main_final = just_main[:num_just_main]
    just_spurious_final = just_spurious[:num_just_spurious]
    neither_final = neither[:num_neither]
    
    ids = []
    for id in both_final:
        ids.append(id)
    for id in just_main_final:
        ids.append(id)
    for id in just_spurious_final:
        ids.append(id)
    for id in neither_final:
        ids.append(id)
        
    # Get the ids of the training images for this experiment
    # By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    ids_train, ids_val = train_test_split(ids, test_size = 0.1)
    
    # Load defaults
    lr = None # This breaks train_model() for configs that aren't setup based on HPS
    mode_param = 0.0
    batch_size = 64
    feature_hook = None

    # Load the the data specified by mode for each Image ID
    if mode in ['initial-transfer', 'initial-tune']:
        names = {}
        names['both'] = {'orig': 1.0}
        names['just_main'] = {'orig': 1.0}
        names['just_spurious'] = {'orig': 1.0}
        names['neither'] = {'orig': 1.0}
        
        if mode == 'initial-transfer':
            lr = 0.001
        elif mode == 'initial-tune':
            lr = 0.0001
        
    elif mode in ['minimal-transfer', 'minimal-tune']:
        if p_correct > 0.5:
            p_sample = 2 - 1 / p_correct
            names = {}
            names['both'] = {'orig': 1.0, 'main-box': p_sample, 'spurious-box': p_sample}
            names['just_main'] = {'orig': 1.0}
            names['just_spurious'] = {'orig': 1.0}
            names['neither'] = {'orig': 1.0}
        elif p_correct < 0.5:
            p_sample = (p_correct - 0.5) / (p_correct - 1)
            names = {}
            names['both'] = {'orig': 1.0}
            names['just_main'] = {'orig': 1.0, 'just_main+just_spurious': p_sample, 'main-box': p_sample}
            names['just_spurious'] = {'orig': 1.0, 'just_spurious+just_main': p_sample, 'spurious-box': p_sample}
            names['neither'] = {'orig': 1.0}
        else:
            print('Error: bad p_correct for this mode')
            sys.exit(0)
        
        if mode == 'minimal-tune':
            lr = 0.0001
        
    elif mode in ['rrr-tune', 'cdep-transfer', 'cdep-tune']:
        name_1 = 'orig'
        name_2 = 'spurious-pixel'
        
        if mode == 'rrr-tune':
            lr = 0.0003
            mode_param = 0.1
        elif mode == 'cdep-transfer':
            lr = 0.003
            mode_param = 1.0
        
    elif mode in ['gs-transfer', 'gs-tune']:
        name_1 = 'orig'
        name_2 = 'main-pixel-paint'
        
        if mode == 'gs-tune':
            lr = 0.00003
            mode_param = 1.0
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
    if mode in ['rrr-tune', 'gs-transfer', 'gs-tune', 'cdep-transfer', 'cdep-tune']:
        files_1_train, labels_1_train, files_2_train = load_data_paired(ids_train, images, name_1, name_2)
        files_1_val, labels_1_val, files_2_val = load_data_paired(ids_val, images, name_1, name_2)
        
        datasets = {}
        datasets['train'] = ImageDataset_Paired(files_1_train, labels_1_train, files_2_train)
        datasets['val'] = ImageDataset_Paired(files_1_val, labels_1_val, files_2_val)
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
    if mode == 'initial-transfer':
        model, optim_params = get_model(mode = 'transfer', parent = 'pretrained')
    elif mode == 'initial-tune':
        model, optim_params = get_model(mode = 'tune', parent = './Models/{}-{}/{}/initial-transfer/trial{}/model.pt'.format(main, spurious, p_correct, trial))
    elif mode in ['minimal-transfer', 'gs-transfer', 'cdep-transfer']:
        model, optim_params = get_model(mode = 'transfer', parent = './Models/{}-{}/{}/initial-tune/trial{}/model.pt'.format(main, spurious, p_correct, trial))
        # Setup the feature hook for getting the representations
        if mode == 'gs-transfer':
            feature_hook = Features(requires_grad = True)
            handle = list(model.modules())[66].register_forward_hook(feature_hook) # Warning:  this is specific to ResNet18
    elif mode in ['minimal-tune', 'rrr-tune', 'gs-tune', 'cdep-tune']:
        model, optim_params = get_model(mode = 'tune', parent = './Models/{}-{}/{}/initial-tune/trial{}/model.pt'.format(main, spurious, p_correct, trial))
    else:
        print('Train.py: Could not determine trainable parameters')
        sys.exit(0)

    model.cuda()

    metric_loss = torch.nn.BCEWithLogitsLoss()
    
    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name,
                        lr_init = lr, select_cutoff = 5, decay_max = 1,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    os.system('rm -rf {}'.format(name)) # Clean up the model history saved during training

if __name__ == '__main__':
    
    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')
    
    for trial in trials:
        train(mode, main, spurious, p_correct, trial)
