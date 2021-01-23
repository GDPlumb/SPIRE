
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import load_data_random, load_data_paired, load_data_fs

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, ImageDataset_Paired, ImageDataset_FS, my_dataloader
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
    
    # Get configuration from mode
    mode_split = mode.split('-')
    
    TRANS = 'transfer' in mode_split
    TUNE = 'tune' in mode_split
    TT = 'tt' in mode_split
    
    INIT = 'initial' in mode_split
    MIN = 'minimal' in mode_split
    SIM = 'simple' in mode_split
    RRR = 'rrr' in mode_split
    CDEP = 'cdep' in mode_split
    GS = 'gs' in mode_split
    FS = 'fs' in mode_split
    
    # Load default parameters
    if TRANS or TT:
        lr = 0.001
    elif TUNE:
        lr = 0.0001
    else:
        print('Error: Could not model ancestry and trainable parameters')
        sys.exit(0)
    
    mode_param = 0.0
    batch_size = 64
    
    # Load the mode specific information
    if INIT:
        names = {}
        names['both'] = {'orig': 1.0}
        names['just_main'] = {'orig': 1.0}
        names['just_spurious'] = {'orig': 1.0}
        names['neither'] = {'orig': 1.0}
        
    elif MIN:
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
            
    elif SIM:
        names = {}
        names['both'] = {'orig': 1.0, 'spurious-box': 1.0}
        names['just_main'] = {'orig': 1.0}
        names['just_spurious'] = {'orig': 1.0}
        names['neither'] = {'orig': 1.0}
        
    elif RRR:
        name_cf = 'spurious-pixel'
        if TUNE:
            mode_param = 10.0
        
    elif CDEP:
        name_cf = 'spurious-pixel'
        if TT:
            mode_param = 1.0
                
    elif GS:
        name_cf = 'main-pixel-paint'
        if TT:
            mode_param = 100.0
        
    elif FS:
        keys_0 = ['just_main', 'neither']
        keys_1 = ['both', 'just_spurious']
        
    else:
        print('Error: Unrecognized mode')
        sys.exit(0)
        
    # Apply parameter overrides
    if mp_override is not None:
        mode_param = mp_override
        
    if lr_override is not None:
        lr = lr_override
        
    if bs_override is not None:
        batch_size = bs_override
    
    # Setup the data loaders
    if RRR or CDEP or GS:
        if not GS:
            files_train, labels_train, files_cf_train, labels_cf_train = load_data_paired(ids_train, images, name_cf)
            files_val, labels_val, files_cf_val, labels_cf_val = load_data_paired(ids_val, images, name_cf)
        else:
            files_train, labels_train, files_cf_train, labels_cf_train = load_data_paired(ids_train, images, name_cf, aug = True)
            files_val, labels_val, files_cf_val, labels_cf_val = load_data_paired(ids_val, images, name_cf, aug = True)
        
        datasets = {}
        datasets['train'] = ImageDataset_Paired(files_train, labels_train, files_cf_train, labels_cf_train)
        datasets['val'] = ImageDataset_Paired(files_val, labels_val, files_cf_val, labels_cf_val)
    elif FS:
        files_train, labels_train, contexts_train = load_data_fs(ids_train, images, splits, keys_0, keys_1)
        files_val, labels_val, contexts_val = load_data_fs(ids_val, images, splits, keys_0, keys_1)

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
    parent_transfer = './Models/{}-{}/{}/initial-transfer/trial{}/model.pt'.format(main, spurious, p_correct, trial)
    parent_tune = './Models/{}-{}/{}/initial-tune/trial{}/model.pt'.format(main, spurious, p_correct, trial)
    if mode == 'initial-transfer':
        model, optim_params = get_model(mode = 'transfer', parent = 'pretrained')
    elif mode == 'initial-tune':
        model, optim_params = get_model(mode = 'tune', parent = parent_transfer)
    elif TRANS:
        model, optim_params = get_model(mode = 'transfer', parent = parent_transfer)
    elif TT:
        model, optim_params = get_model(mode = 'transfer', parent = parent_tune)
    elif TUNE:
        model, optim_params = get_model(mode = 'tune', parent = parent_transfer)
    
    # Setup the feature hook for getting the representations
    # Warning:  this is specific to ResNet18
    if GS and (TRANS or TT):
        feature_hook = Features(requires_grad = True)
        handle = list(model.modules())[66].register_forward_hook(feature_hook)
    elif FS:
        feature_hook = Features()
        handle = list(model.modules())[66].register_forward_hook(feature_hook)
    else:
        feature_hook = None

    # Setup the loss
    if FS:
        metric_loss = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    else:
        metric_loss = torch.nn.BCEWithLogitsLoss()
    
    # Train
    model.cuda()
    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name,
                        lr_init = lr, select_cutoff = 5, decay_max = 1,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    # Clean up the model history saved during training
    os.system('rm -rf {}'.format(name))

if __name__ == '__main__':
    
    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')
    
    for trial in trials:
        train(mode, main, spurious, p_correct, trial)
