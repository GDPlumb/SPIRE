
import json
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import time
import torch

sys.path.insert(0, '../')
from Config import get_data_dir, get_split_sizes

sys.path.insert(0, '../../Common/')
from COCOWrapper import id_from_path
from Dataset import ImageDataset, ImageDataset_Paired, ImageDataset_FS, my_dataloader
from Features import Features
from LoadData import load_images, load_data, load_data_paired, load_data_fs
from Miscellaneous import get_map, get_diff
from ModelWrapper import ModelWrapper
from ResNet import get_model
from TrainModel import train_model, counts_batch, acc_agg

def train(mode, main, spurious, p_correct, trial, 
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
    with open('{}/splits.json'.format(data_dir), 'r') as f:
        splits = json.load(f)
    both = splits['both']
    just_main = splits['just_main']
    just_spurious = splits['just_spurious']
    neither = splits['neither']
    
    # Find the number of images to get from each split
    num_both, num_just_main, num_just_spurious, num_neither = get_split_sizes(p_correct)

    # Select those images
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
    
    PARENT_TRANS = 'ptransfer' in mode_split
    PARENT_TUNE = 'ptune' in mode_split
    
    INIT = 'initial' in mode_split
    AUTO = 'auto' in mode_split
    AUTO_paint = AUTO and 'paint' in mode_split 
    SIM = 'simple' in mode_split
    RRR = 'rrr' in mode_split
    CDEP = 'cdep' in mode_split
    GS = 'gs' in mode_split
    FS = 'fs' in mode_split
    QCEC = 'qcec' in mode_split
    
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
        
    elif AUTO:
        
        if AUTO_paint:
            suffix = 'pixel-paint'
        else:
            suffix = 'box'
        
        # Best config according to HPS:  auto-transfer-ptune
        with open('./FindAugs/{}/probs.json'.format(p_correct), 'r') as f:
            probs = json.load(f)  
        img_types = {}
        img_types['both-main/{}'.format(suffix)] = probs['B2S']
        img_types['both-spurious/{}'.format(suffix)] = probs['B2M']
        img_types['just_main-main/{}'.format(suffix)] = probs['M2N']
        img_types['just_main+spurious'] = probs['M2B']
        img_types['just_spurious-spurious/{}'.format(suffix)] = probs['S2N']
        img_types['just_spurious+main'] = probs['S2B']
        img_types['neither+main'] = probs['N2M']
        img_types['neither+spurious'] = probs['N2S']
        cf_types = [name for name in img_types]
        img_types['orig'] = 1.0
        
    elif SIM:
        img_types = {}
        img_types['both-spurious/box'] = 1.0
        cf_types = [name for name in img_types]
        img_types['orig'] = 1.0
        
    elif RRR:
        cf_types = ['both-spurious/pixel', 'just_spurious-spurious/pixel']
        # Best config according to HPS
        if mode == 'rrr-tune-ptransfer':
            mode_param = 1.0
            
    elif GS:
        cf_types = ['both-main/pixel-paint', 'just_main-main/pixel-paint']
        # Best config according to HPS
        if mode == 'gs-transfer-ptune':
            mode_param = 100.0   
            
    elif CDEP:
        cf_types = ['both-spurious/pixel', 'just_spurious-spurious/pixel']
        # Best config according to HPS
        if mode == 'cdep-transfer-ptune':
            mode_param = 1.0
              
    elif FS:
        cf_types = []
        
        if p_correct < 0.5:
            split_suppress = both_final
            alpha = np.sqrt(num_just_main / num_both)
        elif p_correct > 0.5:
            split_suppress = just_main_final
            alpha = np.sqrt(num_both / num_just_main)
        else:
            print('Error: bad p_correct for this mode')
            sys.exit(0)
        
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
            
    elif QCEC:
        img_types = {}
        img_types['both-main/pixel-paint'] = 0.5
        img_types['both-spurious/pixel-paint'] = 0.5
        img_types['just_main-main/pixel-paint'] = 1.0
        img_types['just_main+spurious'] = 0.0
        img_types['just_spurious-spurious/pixel-paint'] = 1.0
        img_types['just_spurious+main'] = 0.0
        img_types['neither+main'] = 0.0
        img_types['neither+spurious'] = 0.0
        cf_types = [name for name in img_types]
        img_types['orig'] = 1.0
    
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
        
    # Load the required images
    images = load_images(data_dir, cf_types)
    
    # Setup the data loaders
    if INIT or AUTO or SIM or QCEC:
        files_train, labels_train = load_data(ids_train, images, img_types)
        files_val, labels_val = load_data(ids_val, images, img_types)

        datasets = {}
        datasets['train'] = ImageDataset(files_train, labels_train)
        datasets['val'] = ImageDataset(files_val, labels_val) 
        
    elif RRR or CDEP or GS:
        if not GS:
            files_train, labels_train, files_cf_train, labels_cf_train = load_data_paired(ids_train, images, cf_types)
            files_val, labels_val, files_cf_val, labels_cf_val = load_data_paired(ids_val, images, cf_types)
        else:
            files_train, labels_train, files_cf_train, labels_cf_train = load_data_paired(ids_train, images, cf_types, aug = True)
            files_val, labels_val, files_cf_val, labels_cf_val = load_data_paired(ids_val, images, cf_types, aug = True)
        
        datasets = {}
        datasets['train'] = ImageDataset_Paired(files_train, labels_train, files_cf_train, labels_cf_train)
        datasets['val'] = ImageDataset_Paired(files_val, labels_val, files_cf_val, labels_cf_val)
        
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
    parent_trans = './Models/{}-{}/{}/initial-transfer/trial{}/model.pt'.format(main, spurious, p_correct, trial)
    parent_tune = './Models/{}-{}/{}/initial-tune/trial{}/model.pt'.format(main, spurious, p_correct, trial)
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
    if GS and TRANS:
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
    model = train_model(model, optim_params, dataloaders, metric_loss, counts_batch, acc_agg, name = name,
                        lr_init = lr, select_cutoff = 5, decay_max = 1,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    # Clean up the model history saved during training
    os.system('rm -rf {}'.format(name))

def evaluate_acc(model_dir, data_dir):

    # Load the images for this pair
    with open('{}/splits.json'.format(data_dir), 'r') as f:
        splits = json.load(f)
    
    images = load_images(data_dir, [])
        
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir))
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model)
        
    # Run the evaluation
    out = {}
    avg = 0
    for name in ['both', 'just_main', 'just_spurious', 'neither']:
        ids = splits[name]
        files_tmp, labels_tmp = load_data(ids, images, ['orig'])
        
        dataset_tmp = ImageDataset(files_tmp, labels_tmp)
        dataloader_tmp = my_dataloader(dataset_tmp)
        
        y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)
        
        v = np.mean(1 * (y_hat >= 0.5) == y_true)
        out[name] = v
        avg += v
    avg /= 4
    out['average'] = avg
    out['r-gap'] = out['both'] - out['just_main']
    out['h-gap'] = out['neither'] - out['just_spurious']
    
    with open('{}/results.json'.format(model_dir), 'w') as f:
        json.dump(out, f)

def evaluate_cf(model_dir, data_dir):

    # Load the images for this pair
    with open('{}/splits.json'.format(data_dir), 'r') as f:
        splits = json.load(f)
        
    cf_both = ['both-main/box', 'both-main/pixel-paint', 'both-spurious/box', 'both-spurious/pixel-paint']
    cf_main = ['just_main-main/box', 'just_main-main/pixel-paint', 'just_main+spurious']
    cf_spurious = ['just_spurious-spurious/box', 'just_spurious-spurious/pixel-paint', 'just_spurious+main']
    cf_neither = ['neither+main', 'neither+spurious']
    
    cf_types = []
    for cf in [cf_both, cf_main, cf_spurious, cf_neither]:
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
        
    ids = splits['both']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_both:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)
    
    ids = splits['just_main']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_main:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)
        
    ids = splits['just_spurious']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_spurious:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)
        
    ids = splits['neither']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in cf_neither:
        map_name = get_map(wrapper, images, ids, name)
        metrics[name] = get_diff(map_orig, map_name)
        
    with open('{}/counterfactual.json'.format(model_dir), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    
    HPS = int(sys.argv[1]) == 1 # Are we running Hyper Parameter Selection?
    index = sys.argv[2]
    
    # Get the chosen settings
    if not HPS:
        with open('./Models/{}.json'.format(index), 'r') as f:
            configs = json.load(f)
        os.system('rm ./Models/{}.json'.format(index))
    else:
        with open('./HPS/{}.json'.format(index), 'r') as f:
            configs = json.load(f)
        os.system('rm ./HPS/{}.json'.format(index))
    
    for config in configs:
        main = config['main']
        spurious = config['spurious']
        p_correct = config['p_correct']
        mode = config['mode']
        trial = config['trial']
        if HPS:
            mode_param = config['mode_param']
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
        
        if not HPS:
            model_dir = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
        else:
            model_dir = './HPS/{}-{}-{}/{}/{}-{}/trial{}'.format(main, spurious, p_correct, mode, mode_param, learning_rate, trial)

        data_dir = '{}/{}-{}/val'.format(get_data_dir(), main, spurious)
        print(model_dir)
        
        if not HPS:
            train(mode, main, spurious, p_correct, trial, model_dir = model_dir)
        else:
             train(mode, main, spurious, p_correct, trial,
                   lr_override = learning_rate, mp_override = mode_param, bs_override = batch_size,
                   model_dir = model_dir)
        evaluate_acc(model_dir, data_dir)
        evaluate_cf(model_dir, data_dir)

        time.sleep(np.random.uniform(4, 6))
