
import json
import numpy as np
import os
from pathlib import Path
import pickle
import random
from sklearn.model_selection import train_test_split
import sys
import time
import torch

from Config import get_working_dir, get_id, get_splits

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, ImageDataset_FS, my_dataloader
from Features import Features
from MetricUtils import get_accs, get_gaps, get_pr, get_ap
from ModelWrapper import ModelWrapper
from ResNet import get_model
from TrainModel import train_model, counts_batch, fpr_agg


def run_eval(model):
    
    def compare_counterfactuals(wrapper, ids, dataset, cf_type, max_samples = 10000):
        sample = random.sample(ids, min(max_samples, len(ids)))

        # Get predictions on the real image
        filenames_tmp = []
        labels_tmp = []
        for i in sample:
            filenames_tmp.append(dataset[i][0])
            labels_tmp.append(dataset[i][1])

        dataset_tmp = ImageDataset(filenames_tmp, labels_tmp, get_names = True)
        dataloader_tmp = my_dataloader(dataset_tmp)

        y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)

        pred_orig = {}
        for i, v in enumerate(names):
            pred_orig[get_id(v)] = 1 * (y_hat[i][0] >= 0.5)

        # Get predictions on the counterfactual images
        filenames_tmp = []
        labels_tmp = []
        for i in sample:
            filenames_tmp.append(dataset[i][0].replace(get_working_dir(), get_working_dir(cf_type)))
            labels_tmp.append(dataset[i][1])

        dataset_tmp = ImageDataset(filenames_tmp, labels_tmp, get_names = True)
        dataloader_tmp = my_dataloader(dataset_tmp)

        y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)

        pred_cf = {}
        for i, v in enumerate(names):
            pred_cf[get_id(v)] = 1 * (y_hat[i][0] >= 0.5)

        return sum(1 for i in pred_orig if pred_orig[i] != pred_cf[i]) / len(pred_orig)

    def get_split_stats(splits):
        sizes = {}
        n = 0
        for name in splits:
            sizes[name] = len(splits[name])
            n += sizes[name]

        for name in sizes:
            sizes[name] /= n

        B = sizes['both']
        M = sizes['just_main']
        S = sizes['just_spurious']
        N = sizes['neither']

        P_m = B + M
        P_s_m = B / P_m
        P_s_nm = S / (S + N)

        return P_m, P_s_m, P_s_nm

    def get_test_preds(wrapper, splits_in, dataset, max_samples = 10000):
        splits = splits_in.copy()

        # In the ISIC dataset, there are no (or extremely few) examples of 'malignant' images that also have a 'patch'
        # As a result, we are going to evaluate using half of the malignant images normally and then take the other half and add a patch to them
        # By splitting the images this way, we make the model's performance on the two splits indepenent
        splits['both'], splits['just_main'] = train_test_split(splits['just_main'], test_size = 0.5)

        out = {}
        for split in splits:

            ids = splits[split]
            sample = random.sample(ids, min(max_samples, len(ids)))

            filenames_tmp = []
            labels_tmp = []
            for i in sample:
                if split == 'both':
                    filenames_tmp.append(dataset[i][0].replace(get_working_dir(), get_working_dir('add')))
                else:
                    filenames_tmp.append(dataset[i][0])
                labels_tmp.append(dataset[i][1])

            dataset_tmp = ImageDataset(filenames_tmp, labels_tmp, get_names = True)
            dataloader_tmp = my_dataloader(dataset_tmp)

            y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)    
            out[split] = y_hat

        return out
    
    
    out = {}

    # Setup the model
    model.eval()
    wrapper = ModelWrapper(model, get_names = True)

    # Get the training and testing dataset and its splits
    with open('{}/dataset.json'.format(get_working_dir()), 'r') as f:
        dataset = json.load(f)

    splits = {}
    splits['test'] = get_splits('test', 'true')
    splits['train'] = get_splits('train', 'model')
    
    # Counterfactual comparison on the Training Data
    out['cf-add'] = compare_counterfactuals(wrapper, splits['train']['just_main'], dataset['train'], 'add')
    out['cf-remove'] = compare_counterfactuals(wrapper, splits['train']['just_spurious'], dataset['train'], 'remove')
    
    # Testing Set Eval
    P_m, P_s_m, P_s_nm = get_split_stats(splits['test'])
    preds = get_test_preds(wrapper, splits['test'], dataset['test'])
    
    # Convert those predictions to accuracies per threshold per split
    accs = get_accs(preds)
    for name in accs:
        out[name] = accs[name]

    # Use those accuracies to get the gap metrics
    gaps = get_gaps(accs)
    for name in gaps:
        out[name] = gaps[name]
        
    # Use those accuracies to get the precision recall curve and its stats for the balanced distribution
    pr = get_pr(accs, P_m, 0.5, 0.5)
    for name in pr:
        out[name] = pr[name]
    out['ap'] = get_ap(pr)   
        
    # Use those accuracies to get the precision recall curve and its stats for the original distribution
    pr = get_pr(accs, P_m, P_s_m, P_s_nm)
    out['ap-orig'] = get_ap(pr)  
    
    return out

def run(mode, trial,
            mp_override = None, lr_override = None, bs_override = None,
            model_dir = None):
    
    # Setup the output directory
    if model_dir is None:
        model_dir = './Models/{}/trial{}'.format(mode, trial)
    os.system('rm -rf {}'.format(model_dir))
    Path(model_dir).mkdir(parents = True, exist_ok = True)

    name = '{}/model'.format(model_dir)

    # Get configuration from mode
    mode_split = mode.split('-')
        
    INIT = 'initial' in mode_split
    SPIRE = 'spire' in mode_split
    FS = 'fs' in mode_split
    
    TRANS = 'transfer' in mode_split or SPIRE
    TUNE = 'tune' in mode_split or FS
    
    # Split the dataset
    # - By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    # - By setting the random_state with the trial number, we ensure that models in the same trial use the same split
    with open('{}/dataset.json'.format(get_working_dir()), 'r') as f:
        dataset = json.load(f)
    dataset = dataset['train']
    ids = list(dataset)
    
    ids_train, ids_val = train_test_split(ids, test_size = 0.1, random_state = int(trial))

    # Load default parameters
    if TRANS:
        lr = 0.001
    elif TUNE:
        lr = 0.0001
    else:
        print('Error: Could not determine which parameters are to be trained')
        sys.exit(0)
    
    select_cutoff = 3
    decay_max = 1
    select_metric_index = 0
    mode_param = 0.0
    batch_size = 64
    feature_hook = None
    
    # Most models are not using a weighted loss
    if FS:
        metric_loss = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    else:
        metric_loss = torch.nn.BCEWithLogitsLoss()
    
    # Setup the dataloaders
    if INIT:
        dataloaders = {}
        for config in [('train', ids_train), ('val', ids_val)]:
            filenames_tmp = []
            labels_tmp = []
            for i in config[1]:
                filenames_tmp.append(dataset[i][0])
                labels_tmp.append(np.array([dataset[i][1]], dtype = np.float32))

            dataset_tmp = ImageDataset(filenames_tmp, labels_tmp)
            dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
            
    elif SPIRE:
        splits_tmp = get_splits('train', 'model')
        
        dataloaders = {}
        for config in [('train', ids_train), ('val', ids_val)]:
            filenames_tmp = []
            labels_tmp = []
            for i in config[1]:
                id_tmp = get_id(dataset[i][0])
                label_tmp = np.array([dataset[i][1]], dtype = np.float32)
                
                filenames_tmp.append(dataset[i][0])
                labels_tmp.append(label_tmp)
                
                # Load counterfactual versions
                if id_tmp in splits_tmp['both'] or id_tmp in splits_tmp['just_spurious']:
                    filenames_tmp.append('{}/{}.jpg'.format(get_working_dir('remove'), id_tmp))
                    labels_tmp.append(label_tmp)
                
                if id_tmp in splits_tmp['just_main'] or id_tmp in splits_tmp['neither']:
                    filenames_tmp.append('{}/{}.jpg'.format(get_working_dir('add'), id_tmp))
                    labels_tmp.append(label_tmp)
                                                           
            dataset_tmp = ImageDataset(filenames_tmp, labels_tmp)
            dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
            
    elif FS:
        # Ideally, we would suppress the context (ie, patch) for images in Both (ie, Malignant + Patch)
        # But there are no such images
        # After experimenting with what split to suppress, this turned out to be the best configuration
        splits_tmp = get_splits('train', 'model')     
        dataloaders = {}
        for config in [('train', ids_train), ('val', ids_val)]:
            filenames_tmp = []
            labels_tmp = []
            contexts_tmp = []
            for i in config[1]:
                filenames_tmp.append(dataset[i][0])
                labels_tmp.append(np.array([dataset[i][1]], dtype = np.float32))
                context = np.zeros((1), dtype = np.float32)
                if i in splits_tmp['just_main']:
                    context[0] = 1.0
                contexts_tmp.append(context)

            dataset_tmp = ImageDataset_FS(filenames_tmp, labels_tmp, contexts_tmp)
            dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
            
    
    # Setup the model
    parent_trans = './Models/initial-transfer/trial{}/model.pt'.format(trial)
    parent_tune = './Models/initial-tune/trial{}/model.pt'.format(trial)
    if INIT:
        if TRANS:
            model, optim_params = get_model(mode = 'transfer', parent = 'pretrained', out_features = 1)
        elif TUNE:
            model, optim_params = get_model(mode = 'tune', parent = parent_trans, out_features = 1)
    elif SPIRE:
        model, optim_params = get_model(mode = 'transfer', parent = parent_tune, out_features = 1)
    elif FS:
        model, optim_params = get_model(mode = 'tune', parent = parent_tune, out_features = 1)
    model.cuda()
    
    # Setup the feature hook for getting the representations
    # Warning:  this is specific to ResNet18
    if FS:
        feature_hook = Features()
        handle = list(model.modules())[66].register_forward_hook(feature_hook)
    else:
        feature_hook = None    
    
    # Train
    model = train_model(model, optim_params, dataloaders, metric_loss, counts_batch, fpr_agg, 
                        name = name,
                        lr_init = lr, decay_max = decay_max,
                        select_cutoff = select_cutoff, select_metric_index = select_metric_index,
                        mode = mode, mode_param = mode_param, 
                        feature_hook = feature_hook)        
    os.system('rm -rf {}'.format(name))
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    # Eval
    out = run_eval(model)
    with open('{}/results.pkl'.format(model_dir), 'wb') as f:
        pickle.dump(out, f)

if __name__ == '__main__':
     
    index = sys.argv[1]
    
    # Get the chosen settings
    with open('./Models/{}.json'.format(index), 'r') as f:
        configs = json.load(f)
    os.system('rm ./Models/{}.json'.format(index))
    
    for config in configs:
        mode = config['mode']
        trial = config['trial']
        
        model_dir = './Models/{}/trial{}'.format(mode, trial)
        print(model_dir)

        run(mode, trial, model_dir = model_dir)

        time.sleep(np.random.uniform(4, 6))