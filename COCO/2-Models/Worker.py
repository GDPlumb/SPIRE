
from collections import defaultdict
import glob
import json
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import time
import torch

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from COCOWrapper import id_from_path
from Dataset import ImageDataset, ImageDataset_FS, my_dataloader
from Features import Features
from LoadData import load_images, load_data, load_data_fs
from ModelWrapper import ModelWrapper
from ResNet import get_model
from TrainModel import train_model, counts_batch, fpr_agg

def merge(trial):
    model = get_model(mode = 'eval', parent = './Models/initial-tune/trial{}/model.pt'.format(trial), out_features = 91)

    with open('./Categories.json', 'r') as f:
        cats = json.load(f)

    with open('./FindAugs/classes.json', 'r') as f:
        classes = json.load(f)
    
    for i, main in enumerate(classes):
        main = main.replace('+', ' ')
        for cat in cats:
            if cat['name'] == main:
                index = int(cat['id'])
                break

        model_partial = get_model(mode = 'eval', parent = './Models/partial-{}-transfer-ptune/trial{}/model.pt'.format(i, trial), out_features = 91) 
        model.fc.bias[index] = model_partial.fc.bias[index]
        model.fc.weight[index, :] = model_partial.fc.weight[index, :]

    save_dir = './Models/SPIRE/trial{}'.format(trial)
    os.system('rm -rf {}'.format(save_dir))
    Path(save_dir).mkdir(parents = True)
    torch.save(model.state_dict(), '{}/model.pt'.format(save_dir))
    
def train(mode, trial,
            mp_override = None, lr_override = None, bs_override = None,
            model_dir = None):
    
    # Setup the output directory
    if model_dir is None:
        model_dir = './Models/{}/trial{}'.format(mode, trial)
    os.system('rm -rf {}'.format(model_dir))
    Path(model_dir).mkdir(parents = True, exist_ok = True)
    
    name = '{}/model'.format(model_dir)

    # Get the ids of the training images for this experiment
    # By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    data_dir = '{}/train'.format(get_data_dir())
    images = load_images(data_dir, [])
    ids = [key for key in images]
    ids_train, ids_val = train_test_split(ids, test_size = 0.1)
    
    # Get configuration from mode
    mode_split = mode.split('-')
    
    TRANS = 'transfer' in mode_split
    TUNE = 'tune' in mode_split
    
    PARENT_TRANS = 'ptransfer' in mode_split
    PARENT_TUNE = 'ptune' in mode_split
    
    INIT = 'initial' in mode_split
    PART = 'partial' in mode_split
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
    
    # Get the category/label info
    with open('./Categories.json', 'r') as f: # This is a json copy of coco.loadCats(coco.getCatIds())
        cats = json.load(f)
    cats_chosen = cats
    
    # Load the mode specific information
    if INIT:
        img_types = {}
        cf_types = []
        img_types['orig'] = 1.0
    
    elif PART:
        # Get the name of Main
        i = int(mode_split[1])
        with open('./FindAugs/classes.json', 'r') as f:
            main = json.load(f)[i]
            
        # Get the index for Main
        for cat in cats:
            if cat['name'] == main.replace('+', ' '):
                index = int(cat['id'])
                cats_chosen = [cat] # Overwrite:  Only eval on this category
                break

        # Get the sampling probabilities for this augmentation
        with open('./FindAugs/{}/names.json'.format(main), 'r') as f:
            img_types = json.load(f)
        
        cf_types = [name for name in img_types]
        cf_types.remove('orig')
        
    elif FS:
        cf_types = []
        
        with open('../0-FindPairs/Pairs.json', 'r') as f:
            pairs = json.load(f)

        id2info = defaultdict(list)
        for pair in pairs:
            main = pair.split('-')[0].replace('+', ' ')
            index = None
            for cat in cats:
                if cat['name'] == main:
                    index = int(cat['id'])

            with open('{}/train/splits/{}.json'.format(get_data_dir(), pair)) as f:
                splits = json.load(f)

            num_both = len(splits['both'])
            num_main = len(splits['just_main'])

            if num_both >= num_main:
                split_suppress = 'just_main'
                alpha = np.sqrt(num_both / num_main)
            else:
                split_suppress = 'both'
                alpha = np.sqrt(num_main / num_both)
                
            # Apply the override early because this parameter is not passed to the training function
            if mp_override is not None:
                mode_param = mp_override
            # Best config according to HPS
            elif mode == 'fs-tune-ptune':
                mode_param = 1000.0
            if alpha < mode_param:
                alpha = mode_param

            for id in splits[split_suppress]:
                info = (index, alpha)
                id2info[id].append(info)
            
    else:
        print('Error: Unrecognized mode')
        sys.exit(0)
        
    # Setup the label indices used
    indices = [int(cat['id']) for cat in cats_chosen]
    
    def counts_batch_cust(y_hat, y):
        return counts_batch(y_hat, y, indices = indices)
        
    # Apply parameter overrides
    if mp_override is not None:
        mode_param = mp_override
        
    if lr_override is not None:
        lr = lr_override
        
    if bs_override is not None:
        batch_size = bs_override
        
    # Load the required images
    images = load_images(data_dir, cf_types)
        
    # Setup the data loaders
    if INIT or PART:
        files_train, labels_train = load_data(ids_train, images, img_types, indices_preserve = indices)
        files_val, labels_val = load_data(ids_val, images, img_types, indices_preserve = indices)

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
    parent_trans = './Models/initial-transfer/trial{}/model.pt'.format(trial)
    parent_tune = './Models/initial-tune/trial{}/model.pt'.format(trial)
    if mode == 'initial-transfer':
        model, optim_params = get_model(mode = 'transfer', parent = 'pretrained', out_features = 91)
    elif mode == 'initial-tune':
        model, optim_params = get_model(mode = 'tune', parent = parent_trans, out_features = 91)
    elif TRANS:
        if PARENT_TRANS:
            model, optim_params = get_model(mode = 'transfer', parent = parent_trans, out_features = 91)
        elif PARENT_TUNE:
            model, optim_params = get_model(mode = 'transfer', parent = parent_tune, out_features = 91)
    elif TUNE:
        if PARENT_TRANS:
            model, optim_params = get_model(mode = 'tune', parent = parent_trans, out_features = 91)
        elif PARENT_TUNE:
            model, optim_params = get_model(mode = 'tune', parent = parent_tune, out_features = 91)
            
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
    model = train_model(model, optim_params, dataloaders, metric_loss, counts_batch_cust, fpr_agg, name = name,
                        lr_init = lr, select_cutoff = 3, decay_max = 1,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    # Clean up the model history saved during training
    os.system('rm -rf {}'.format(name))

def evaluate(model_dir, data_dir, min_samples = 25):

    # Load the needed information
    images = load_images(data_dir, [])
    
    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)
        
    with open('./Categories.json', 'r') as f: # This is a json copy of coco.loadCats(coco.getCatIds())
        cats = json.load(f)
    
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir), out_features = 91)
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    
    # Get the predictions for all of dataset
    ids = [id for id in images]
    files, labels = load_data(ids, images, ['orig'])
    dataset = ImageDataset(files, labels, get_names = True)
    dataloader = my_dataloader(dataset)
    y_hat, y_true, names = wrapper.predict_dataset(dataloader)
    
    data_map = {}
    for i, name in enumerate(names):
        data_map[id_from_path(name)] = [y_hat[i], y_true[i]]
                
    # Run the evaluation
    out = {}
    
    precision, recall = wrapper.metrics(y_hat, y_true)
    MAP = 0.0
    MAR = 0.0
    for cat in cats:
        p = precision[cat['id']]
        MAP += p
        r = recall[cat['id']]
        MAR += r
    MAP /= len(cats)
    MAR /= len(cats)
    out['MAP'] = MAP
    out['MAR'] = MAR
    
    for pair in pairs:
        main = pair.split('-')[0]
        spurious = pair.split('-')[1]
        
        # Get the index that we care about for this pair
        for cat in cats:
            if cat['name'] == main.replace('+', ' '):
                index = int(cat['id'])
                break
        
        # Get the image splits for this pair
        with open('{}/splits/{}-{}.json'.format(data_dir, main, spurious), 'r') as f:
            splits = json.load(f)
        
        # Calculate the accuracy for each split
        splits_acc = {}
        for split_name in splits:
            split = splits[split_name]
            n = len(split)

            pred = np.zeros((n))
            true = np.zeros((n))
            for i, id in enumerate(split):
                data_tmp = data_map[id]
                pred[i] = data_tmp[0][index]
                true[i] = data_tmp[1][index]
            
            if len(pred) >= min_samples:
                v = np.mean(1 * (pred >= 0.5) == true)
            else:
                # Load the ChallengeSet
                files_cs = glob.glob('../0-FindPairs/ExternalData/{}-{}/*'.format(pair, split_name))
                n_cs = len(files_cs)
                if n_cs == 0:
                    print('Error:  missing external data - ', pair, split_name)
                    
                if split_name in ['both', 'just_main']:
                    labels_cs = np.ones((n_cs), dtype = np.float32)
                elif split_name in ['just_spurious', 'neither']:
                    labels_cs = np.zeros((n_cs), dtype = np.float32)
                
                dataset_cs = ImageDataset(files_cs, labels_cs, get_names = True)
                dataloader_cs = my_dataloader(dataset_cs)
                y_hat_cs, y_true_cs, names_cs = wrapper.predict_dataset(dataloader_cs)
                v = np.mean((y_hat_cs[:, index] >= 0.5) == y_true_cs)
            
            out['{}-{}'.format(pair, split_name)] = v
            splits_acc[split_name] = v
        
        # Get some common information for computing metrics
        for key in splits:
            splits[key] = len(splits[key])
        total = splits['both'] + splits['just_main'] + splits['just_spurious'] + splits['neither']
        p_main = (splits['both'] + splits['just_main']) / total
        
        both = splits_acc['both']
        just_main = splits_acc['just_main']
        just_spurious = splits_acc['just_spurious']
        neither = splits_acc['neither']

        # Compute the 'balanced' precision and recall
        # -  We keep P(Main) from the original dataset
        # -  We then construct a distribution where Spurious is independent from Main
        # -  We use either 1/2 or P(Spurious) from the original dataset while doing this
        p_spurious = 0.5 #(splits['both'] + splits['just_spurious']) / total
        tp = p_main * (p_spurious * both + (1 - p_spurious) * just_main)
        fp = (1 - p_main) * (p_spurious * (1 - just_spurious)  + (1 - p_spurious) * (1 -  neither))
        precision = tp / (tp + fp + 1e-16)
        out['{}-b-precision'.format(pair)] = precision
        out['{}-b-recall'.format(pair)] = p_spurious * both + (1 - p_spurious) * just_main
        
        # Compute the "gap" metrics
        out['{}-r-gap'.format(pair)] = both - just_main
        out['{}-h-gap'.format(pair)] = neither - just_spurious

    with open('{}/results.json'.format(model_dir), 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':
     
    SPIRE = sys.argv[1]  
    index = sys.argv[2]
    
    TRAIN = SPIRE in ['0', '1']
    EVAL = SPIRE in ['0', '2']
    
    # Get the chosen settings
    with open('./Models/{}.json'.format(index), 'r') as f:
        configs = json.load(f)
    os.system('rm ./Models/{}.json'.format(index))
    
    for config in configs:
        mode = config['mode']
        trial = config['trial']
        
        model_dir = './Models/{}/trial{}'.format(mode, trial)
        data_dir = '{}/val'.format(get_data_dir())
        print(model_dir)

        if TRAIN:
            train(mode, trial, model_dir = model_dir)
        if EVAL:
            evaluate(model_dir, data_dir)

        time.sleep(np.random.uniform(4, 6))
