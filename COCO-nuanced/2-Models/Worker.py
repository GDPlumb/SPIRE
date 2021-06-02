
import glob
import json
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import sys
import time
import torch

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from Dataset import ImageDataset, ImageDataset_FS, my_dataloader
from Features import Features
from LoadData import load_images, load_data, load_data_fs
from Miscellaneous import get_map, get_diff
from ModelWrapper import ModelWrapper
from ResNet import get_model
from TrainModel import train_model, counts_batch, fpr_agg, acc_agg

def train(mode, label1, label2, spurious, trial,
            mp_override = None, lr_override = None, bs_override = None,
            model_dir = None):

    # Setup the output directory
    if model_dir is None:
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(label1, label2, spurious, mode, trial)
    os.system('rm -rf {}'.format(model_dir))
    Path(model_dir).mkdir(parents = True, exist_ok = True)

    name = '{}/model'.format(model_dir)
        
    # Load the chosen images for this tuple
    data_dir = '{}/{}-{}/{}/train'.format(get_data_dir(), label1, label2, spurious )
    with open('{}/splits.json'.format(data_dir), 'rb') as f:
        splits = json.load(f)
    
    ids = []
    for key in splits:
        for id in splits[key]:
            ids.append(id)
        
    # Get the ids of the training images for this experiment
    # By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    ids_train, ids_val = train_test_split(ids, test_size = 0.1)
    
    # Get configuration from mode
    mode_split = mode.split('-')
    
    TRANS = 'transfer' in mode_split
    TUNE = 'tune' in mode_split
    
    PARENT_TRANS = 'ptransfer' in mode_split
    PARENT_TUNE = 'ptune' in mode_split
    
    INIT = 'initial' in mode_split
    COM = 'combined' in mode_split
    REM = 'removed' in mode_split
    ADD = 'added' in mode_split
    FS = 'fs' in mode_split

    # Load default parameters
    if TRANS:
        lr = 0.001
    elif TUNE:
        lr = 0.0001
    else:
        print('Error: Could not determine which parameters are to be trained')
        sys.exit(0)
    
    mode_param = 0.0
    batch_size = 64

    # Load the mode specific information
    if INIT:
        img_types = {}
        cf_types = []
        img_types['orig'] = 1.0
     
    elif COM:
        img_types = {}
        img_types['1s-spurious/box'] = 1.0
        img_types['0s-spurious/box'] = 1.0
        img_types['1ns+spurious'] = 1.0
        img_types['0ns+spurious'] = 1.0
        cf_types = [name for name in img_types]
        img_types['orig'] = 1.0
        
    elif REM:
        img_types = {}
        img_types['1s-spurious/box'] = 1.0
        img_types['0s-spurious/box'] = 1.0
        cf_types = [name for name in img_types]
        img_types['orig'] = 1.0
        
    elif ADD:
        img_types = {}
        img_types['1ns+spurious'] = 1.0
        img_types['0ns+spurious'] = 1.0
        cf_types = [name for name in img_types]
        img_types['orig'] = 1.0

    elif FS:
        cf_types = []
        
        num_1s = len(splits['1s'])
        num_1ns = len(splits['1ns'])

        if num_1s >= num_1ns:
            split_suppress = splits['1ns']
            alpha = np.sqrt(num_1s / num_1ns)
        else:
            split_suppress = splits['1s']
            alpha = np.sqrt(num_1ns / num_num_1s)
        
        # Apply the override early because this parameter is not passed to the training function
        if mp_override is not None:
            mode_param = mp_override
        # Best config according to HPS
        elif mode == 'fs-tune-ptune':
            mode_param = 1000.0
        if alpha < mode_param:
            alpha = mode_param

        id2info = {}
        for id in split_suppress:
            id2info[id] = [(0, alpha)]
    
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
        
    # Load the required images
    images = load_images(data_dir, cf_types)
    
    # Setup the data loaders
    if INIT or COM or REM or ADD:
        files_train, labels_train = load_data(ids_train, images, img_types)
        files_val, labels_val = load_data(ids_val, images, img_types)

        datasets = {}
        datasets['train'] = ImageDataset(files_train, labels_train)
        datasets['val'] = ImageDataset(files_val, labels_val)
        
    elif FS:
        files_train, labels_train, contexts_train = load_data_fs(ids_train, images, id2info)
        files_val, labels_val, contexts_val = load_data_fs(ids_val, images, id2info)

        datasets = {}
        datasets['train'] = ImageDataset_FS(files_train, labels_train, contexts_train)
        datasets['val'] = ImageDataset_FS(files_val, labels_val, contexts_val)

    dataloaders = {}
    dataloaders['train'] = my_dataloader(datasets['train'], batch_size = batch_size)
    dataloaders['val'] = my_dataloader(datasets['val'], batch_size = batch_size)
    
    # Setup the model and optimization process
    parent_trans = './Models/{}-{}/{}/initial-transfer/trial{}/model.pt'.format(label1, label2, spurious, trial)
    parent_tune = './Models/{}-{}/{}/initial-tune/trial{}/model.pt'.format(label1, label2, spurious, trial)
    if mode == 'initial-transfer':
        model, optim_params = get_model(mode = 'transfer', parent = 'pretrained')
    elif mode == 'initial-tune':
        model, optim_params = get_model(mode = 'tune', parent = parent_trans)
    elif TRANS:
        if PARENT_TRANS:
            model, optim_params = get_model(mode = 'transfer', parent = parent_trans)
        elif PARENT_TUNE:
            model, optim_params = get_model(mode = 'transfer', parent = parent_tune)
    elif TUNE:
        if PARENT_TRANS:
            model, optim_params = get_model(mode = 'tune', parent = parent_trans)
        elif PARENT_TUNE:
            model, optim_params = get_model(mode = 'tune', parent = parent_tune)
    
    # Setup the feature hook for getting the representations
    # Warning:  this is specific to ResNet18
    if FS:
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
    model = train_model(model, optim_params, dataloaders, metric_loss, counts_batch, fpr_agg, name = name,
                            lr_init = lr, select_cutoff = 5, decay_max = 1,
                            mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    # Clean up the model history saved during training
    os.system('rm -rf {}'.format(name)) 

# The following 3 functions are based on COCO/2-Models/Worker.py
def get_accs(preds, num = 101):
    thresholds = np.linspace(0, 1, num = num)
    
    accs = {}
    for name in preds:
        POS = None
        if '1' in name:
            POS = True
        elif '0' in name:
            POS = False
        else:
            print('Warning:  bad name')
        
        p = preds[name]
        n = len(p)
        p = np.sort(p, axis = 0)        
        
        result = np.zeros((num))
        index = 0
        for i, t in enumerate(thresholds):
            while index < n and p[index] < t:
                index += 1
            if POS:
                result[i] = 1 - index / n
            else:
                result[i] = index / n
        
        accs[name] = result
    
    return accs    

def get_gaps(accs, num = 101):
    thresholds = np.linspace(0, 1, num = num)
        
    r_gap = np.abs(accs['1s'] - accs['1ns'])
    h_gap = np.abs(accs['0s'] - accs['0ns'])
    
    out = {}
    out['r-gap'] = r_gap
    out['h-gap'] = h_gap
    return out

def get_pr(accs, P_1, P_s_1, P_s_0):
    tp = P_1 * (P_s_1 * accs['1s'] + (1 - P_s_1) * accs['1ns'])
    fp = (1 - P_1) * (P_s_0 * (1 - accs['0s']) + (1 - P_s_0) * (1 - accs['0ns'])) 
        
    recall = tp / P_1
    precision = tp / (tp + fp + 1e-16)
    precision[np.where(tp == 0.0)] = 1.0
        
    out = {}
    out['precision'] = precision
    out['recall'] = recall
    return out
    
def evaluate_acc(model_dir, data_dir, min_size = 25, challenge_info = None):

    # Load the images for this pair
    with open('{}/splits.json'.format(data_dir), 'rb') as f:
        splits = json.load(f)
    
    images = load_images(data_dir, [])
        
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir))
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model)
        
    # Run the evaluation
    out = {}
    for name in ['1s', '1ns', '0s', '0ns']:
        ids = splits[name]
        files_tmp, labels_tmp = load_data(ids, images, ['orig'])
        
        if len(files_tmp) < min_size:
            v = -1
        else:
            dataset_tmp = ImageDataset(files_tmp, labels_tmp)
            dataloader_tmp = my_dataloader(dataset_tmp)
            
            y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)
            
            v = np.mean(1 * (y_hat >= 0.5) == y_true)
        out['orig-{}'.format(name)] = v

    # Run the Challenge Set evaluation
    if challenge_info is not None:
        label1 = challenge_info[0]
        label2 = challenge_info[1]
        spurious = challenge_info[2]
        
        preds = {}
        configs = [('{}+{}'.format(label1, spurious), 1, '1s'), \
                   ('{}-{}'.format(label1, spurious), 1, '1ns'), \
                   ('{}+{}'.format(label2, spurious), 0, '0s'), \
                   ('{}-{}'.format(label2, spurious), 0, '0ns')]
        for config in configs:
            folder = config[0]
            label = config[1]
            name = config[2]

            files_tmp = []
            labels_tmp = []
            for file in glob.glob('../0-FindTuples/ExternalData/{}/*'.format(folder)):
                files_tmp.append(file)
                labels_tmp.append(label)

            dataset_tmp = ImageDataset(files_tmp, labels_tmp)
            dataloader_tmp = my_dataloader(dataset_tmp)

            y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)
            preds[name] = y_hat
        
        accs = get_accs(preds)
        for name in accs:
            out[name] = accs[name]
            
        info = get_gaps(accs)
        for name in info:
            out[name] = info[name]
        
        P_1 = (len(splits['1s']) + len(splits['1ns'])) / (len(splits['1s']) + len(splits['1ns']) + len(splits['0s']) + len(splits['0ns']))
        
        info = get_pr(accs, P_1, 0.5, 0.5)
        for name in info:
            out[name] = info[name]
    
    with open('{}/results.pkl'.format(model_dir), 'wb') as f:
        pickle.dump(out, f)
        
def evaluate_cf(model_dir, data_dir):

    # Load the images for this pair
    with open('{}/splits.json'.format(data_dir), 'rb') as f:
        splits = json.load(f)

    cf_1s = ['1s-spurious/box', '1s-spurious/pixel-paint']
    cf_1ns = ['1ns+spurious']
    cf_0s = ['0s-spurious/box', '0s-spurious/pixel-paint']
    cf_0ns = ['0ns+spurious']
    
    cf_types = []
    for cf in [cf_1s, cf_1ns, cf_0s, cf_0ns]:
        for v in cf:
            cf_types.append(v)
           
    images = load_images(data_dir, cf_types)
        
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir))
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    
    # Get the model's predictions on each images split
    metrics = {}
        
    ids = splits['1s']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_1s:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)
        
    ids = splits['1ns']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_1ns:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)

    ids = splits['0s']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_0s:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)

    ids = splits['0ns']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_0ns:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)
        
    with open('{}/counterfactual.json'.format(model_dir), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    
    # Get the chosen settings
    index = sys.argv[1]
    with open('./Models/{}.json'.format(index), 'r') as f:
        configs = json.load(f)
    os.system('rm ./Models/{}.json'.format(index))
    
    for config in configs:
        label1 = config['label1']
        label2 = config['label2']
        spurious = config['spurious']
        mode = config['mode']
        trial = config['trial']
        
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(label1, label2, spurious, mode, trial)
        data_dir = '{}/{}-{}/{}/val'.format(get_data_dir(), label1, label2, spurious)
        print(model_dir)
        
        train(mode, label1, label2, spurious, trial, model_dir = model_dir)
        evaluate_acc(model_dir, data_dir, challenge_info = (label1, label2, spurious))
        evaluate_cf(model_dir, data_dir)
        
        time.sleep(np.random.uniform(4, 6))
