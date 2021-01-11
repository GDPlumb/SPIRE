
import json
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import torch

from Config import get_data_dir, get_data_fold

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, my_dataloader
from LoadData import load_data
from ResNet import get_model
from TrainModel import train_model

with open('./COCO_cats.json', 'r') as f: #This is a json copy of coco.loadCats(coco.getCatIds())
    cats = json.load(f)

def get_counts(y_hat, y):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y[i] == y_hat[i] == 1:
           TP += 1
        if y_hat[i] == 1 and y[i] == 0:
           FP += 1
        if y[i] == y_hat[i] == 0:
           TN += 1
        if y_hat [i] == 0 and y[i] == 1:
           FN += 1

    return [TP, FP, TN, FN]

def metric_acc_batch(y_hat, y, cats = cats):
    y_hat = y_hat.cpu().data.numpy()
    y_hat = 1 * (y_hat >= 0)
    y = y.cpu().data.numpy()
    
    out = np.zeros((len(cats), 4))
    c = 0
    for cat in cats:
        index = cat['id']
        # BUG:  The 'squeeze' here might cause this to crash for a batch size of 1
        out[c, :] = get_counts(y_hat[:, index], y[:, index]) #get_counts(np.squeeze(y_hat[:, index]), np.squeeze(y[:, index]))
        c += 1
     
    return out
    
def metric_acc_agg(counts_list = None):
    if counts_list is None:
        return ['F1', 'Precision', 'Recall']
    else:
        counts_agg = sum(counts_list)
        num_classes = counts_agg.shape[0]
        
        precision = np.zeros((num_classes))
        recall = np.zeros((num_classes))
        f1 = np.zeros((num_classes))
        for i in range(num_classes):
            counts = counts_agg[i, :]
            precision[i] = counts[0] / max((counts[0] + counts[1]), 1)
            recall[i] = counts[0] / max((counts[0] + counts[3]), 1)
            if precision[i] != 0 and recall[i] != 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = 0

        return [np.mean(f1), np.mean(precision), np.mean(recall)]
            
def train(mode, trial,
            mp_override = None, lr_override = None, bs_override = None,
            model_dir = None):
    
    if model_dir is None:
        model_dir = './Models/{}/trial{}'.format(mode, trial)
    os.system('rm -rf {}'.format(model_dir))
    Path(model_dir).mkdir(parents = True, exist_ok = True)
    
    name = '{}/model'.format(model_dir)

    # Load the data
    df = get_data_fold()
    if df == -1:
        fold = 'val'
    else:
        fold = 'train'
    
    data_dir = '{}/{}'.format(get_data_dir(), fold)
    with open('{}/images.json'.format(data_dir), 'r') as f:
        images = json.load(f)

    # Get the ids of the training images for this experiment
    # By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    ids = [key for key in images]
    ids_train, ids_val = train_test_split(ids, test_size = 0.1)

    # Load defaults
    if 'transfer' in mode.split('-'):
        lr = 0.001
    elif 'tune' in mode.split('-'):
        lr = 0.0001
    else:
        lr = None
    
    select_cutoff = 3
    mode_param = 0.0
    batch_size = 64
    feature_hook = None
    indices_preserve = None
    
    # Load the the data specified by mode for each Image ID
    if mode in ['initial-transfer', 'initial-tune']:
        names = ['orig']
    elif mode.split('-')[0] == 'partial':
        
        select_cutoff = 1
        
        with open('./FindAugs/classes.json', 'r') as f:
            mains = json.load(f)
            
        i = int(mode.split('-')[1])
        main = mains[i]
        
        with open('./FindAugs/{}/names.json'.format(main), 'r') as f:
            names = json.load(f)
        
        for cat in cats:
            if cat['name'] == main.replace('+', ' '):
                index = int(cat['id'])
                break
        
        indices_preserve = [index] # Zero out all of the other labels to stop the model from learning using them

    else:
        print('Error: Unrecognized mode')
        sys.exit(0)
        
    # Setup the data loaders
    if mode in []:
        pass # Used for methods that pair the real and counterfactual examples
    else:
        files_train, labels_train = load_data(ids_train, images, names, indices_preserve = indices_preserve)
        files_val, labels_val = load_data(ids_val, images, names, indices_preserve = indices_preserve)

        datasets = {}
        datasets['train'] = ImageDataset(files_train, labels_train)
        datasets['val'] = ImageDataset(files_val, labels_val)

    dataloaders = {}
    dataloaders['train'] = my_dataloader(datasets['train'], batch_size = batch_size)
    dataloaders['val'] = my_dataloader(datasets['val'], batch_size = batch_size)

    # Setup the model and optimization process
    parent_transfer = './Models/initial-transfer/trial{}/model.pt'.format(trial)
    if mode == 'initial-transfer':
        model, optim_params = get_model(mode = 'transfer', parent = 'pretrained', out_features = 91)
    elif mode in ['initial-tune']:
        model, optim_params = get_model(mode = 'tune', parent = parent_transfer, out_features = 91)
    elif mode.split('-')[0] == 'partial':
        model, optim_params = get_model(mode = 'transfer', parent = './Models/initial-tune/trial{}/model.pt'.format(trial), out_features = 91)
    else:
        print('Error: Could not determine trainable parameters')
        sys.exit(0)

    model.cuda()

    metric_loss = torch.nn.BCEWithLogitsLoss()

    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name,
                        lr_init = lr, select_cutoff = select_cutoff, decay_max = 1,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    os.system('rm -rf {}'.format(name)) # Clean up the model history saved during training
