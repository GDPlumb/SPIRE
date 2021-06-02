
from collections import defaultdict
import glob
import json
import numpy as np
import os
from pathlib import Path
import pickle
import random
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import sys
import time
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from COCOWrapper import id_from_path
from Dataset import ImageDataset, ImageDataset_FS, my_dataloader
from LoadData import load_ids, load_data_fs
from ModelWrapper import ModelWrapper
from ResNet import get_model, get_features, get_lm, set_lm
from TrainModel import train_model, counts_batch, fpr_agg

def get_representation(model, filenames, labels):
    # Setup the model
    model.cuda()
    model.eval()
    
    # Get the feature layer for the model
    feature_hook = get_features(model)
    
    # Find the model's representations
    dataset = ImageDataset(filenames, labels, get_names = True)
    dataloader = my_dataloader(dataset)
    
    data = {}
    for batch in dataloader:
        x = batch[0].cuda()
        y = batch[1].numpy()
        f = batch[2]

        y_hat = model(x)
        rep = feature_hook.features.data.cpu().numpy()[:, :, 0, 0]

        for i in range(len(y_hat)):
            data[id_from_path(f[i])] = [rep[i, :], y[i, :]]

    return data

def predict(model, config):
    # Get the config
    # If we are running HPS, this loads a large (unused) chunk of the training dataset
    # Otherwise, it loads the validation dataset and enables us to check the external datasets
    images = config[0]
    ids = config[1]
    external = config[2]

    # Setup the model
    model.eval()
    model.cuda()
    wrapper = ModelWrapper(model, get_names = True)
    
    # Get the predictions for all of the original dataset
    files, labels = load_ids(ids, images)
    labels = np.array(labels, dtype = np.float32)
    dataset = ImageDataset(files, labels, get_names = True)
    dataloader = my_dataloader(dataset)
    y_hat, y_true, names = wrapper.predict_dataset(dataloader)
    
    preds_orig = {}
    for i, name in enumerate(names):
        preds_orig[id_from_path(name)] = y_hat[i]
        
    out = {}
    out['orig'] = preds_orig
                
    # Get the predictions on the external data
    if external:
        for name in glob.glob('../0-FindPairs/ExternalData/*'):
            files_ext = glob.glob('{}/*'.format(name))
            n_ext = len(files_ext)
            labels_ext = -1 * np.ones((n_ext))
            dataset_ext = ImageDataset(files_ext, labels_ext, get_names = True)
            dataloader_ext = my_dataloader(dataset_ext)
            y_hat_ext, y_true_ext, names_ext = wrapper.predict_dataset(dataloader_ext)
            
            out[name.split('/')[-1]] = y_hat_ext

    return out

def get_index(pair):
    main = pair.split('-')[0].replace('+', ' ')
    
    with open('./Categories.json', 'r') as f:
        cats = json.load(f)
        
    for cat in cats:
        if cat['name'] == main:
            index = int(cat['id'])
            break
            
    return index

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

    P_m = B + M # P(Main)
    bias = B / (B + M) - (B + S) #P(Spurious | Main) - P(Spurious)
    
    out = {}
    out['P_m'] = P_m
    out['bias'] = bias
    
    # Calculate the upper bound for SPIRE's sampling probability
    # - Tries to set P(Spurious | Main) = P(Spurious | not Main)
    if bias >= 0:      
        a = 1
        b = M + S
        c = M * S - B * N

        delta = (-1.0 * b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

        if delta < 0:
            delta = (-1.0 * b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

        if delta > B:
            delta = B

        out['s_p1'] = (delta / B, 'both-main/box')
        out['s_p2'] = (delta / B, 'both-spurious/box')

    else:
        delta = (B * N - M * S) / (M - N)

        if delta > M:
            delta = M

        out['s_p1'] = (delta / M, 'just_main+spurious')
        out['s_p2'] = (delta / N, 'neither+spurious')

    return out

def get_accs(preds, num = 101):
    thresholds = np.linspace(0, 1, num = num)
    
    accs = {}
    for name in preds:
        POS = None
        if name in ['both', 'just_main']:
            POS = True
        elif name in ['just_spurious', 'neither']:
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
        
    r_gap = np.abs(accs['both'] - accs['just_main'])
    h_gap = np.abs(accs['just_spurious'] - accs['neither'])
    
    out = {}
    out['r-gap'] = r_gap
    out['h-gap'] = h_gap
    return out

def get_pr(accs, P_m, P_s_m, P_s_nm):
    tp = P_m * (P_s_m * accs['both'] + (1 - P_s_m) * accs['just_main'])
    fp = (1 - P_m) * (P_s_nm * (1 - accs['just_spurious']) + (1 - P_s_nm) * (1 - accs['neither'])) 
        
    recall = tp / P_m
    precision = tp / (tp + fp + 1e-16)
    precision[np.where(tp == 0.0)] = 1.0
        
    out = {}
    out['precision'] = precision
    out['recall'] = recall
    return out

def interpolate(x, y, x_t):
    y_rev = list(reversed(list(y)))
    x_rev = list(reversed(list(x)))
    return np.interp(x_t, x_rev, y_rev)

def get_metrics(pair, preds, index = None, data_split = 'val', max_samples = None):
    # Get the index of the main object    
    if index is None:
        index = get_index(pair)

    # Get the splits
    with open('{}/{}/splits/{}.json'.format(get_data_dir(), data_split, pair), 'r') as f:
        splits = json.load(f)

    info = get_split_stats(splits)
    P_m = info['P_m']
    
    # Get the predictions for each split
    orig = preds['orig']
    ids = list(orig)    
    preds_pair = defaultdict(list)
    for name in splits:
        # On original images
        if max_samples is not None:
            size = min(max_samples, len(splits[name]))
            ids_split = np.intersect1d(random.sample(splits[name], size), ids)
        else:           
            ids_split = np.intersect1d(splits[name], ids)
            
        for i in ids_split:
            preds_pair[name].append(orig[i][index]) 

        # On external images
        name_ext = '{}-{}'.format(pair, name)
        if name_ext in preds:
            for y in preds[name_ext]:
                preds_pair[name].append(y[index])
    
    out = {}
    
    # Convert those predictions to accuracies per threshold per split
    accs = get_accs(preds_pair)
    for name in accs:
        out[name] = accs[name]

    # Use those accuracies to get the gap metrics
    info = get_gaps(accs)
    for name in info:
        out[name] = info[name]
        
    # Use those accuracies to get the precision recall curve and its stats for the balanced distribution
    info = get_pr(accs, P_m, 0.5, 0.5)
    for name in info:
        out[name] = info[name]
    
    thresholds = np.linspace(0, 1, num = 101)
    pr_curve = interpolate(out['recall'], out['precision'], thresholds)
    out['ap'] = auc(thresholds, pr_curve)
    
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
    
    HPS = 'hps' in mode_split
    
    INIT = 'initial' in mode_split
    SPIRE = 'spire' in mode_split
    FS = 'fs' in mode_split
    
    TRANS = 'transfer' in mode_split or SPIRE
    TUNE = 'tune' in mode_split or FS
        
    # Split the dataset
    # - By splitting on Image ID, we ensure all counterfactual version of an image are in the same fold
    # - By setting the random_state with the trial number, we ensure that models in the same trial use the same split
    with open('{}/train/images.json'.format(get_data_dir()), 'r') as f:
        images = json.load(f)
    ids = list(images)
    if HPS:
        ids_train, ids_val_all = train_test_split(ids, test_size = 0.5, random_state = int(trial))
        ids_val = random.sample(ids_val_all, int(0.5 * len(ids_train)))
        
        def get_eval_config():
            images_eval = images
            ids_eval = ids_val_all
            external_eval = False
            return images_eval, ids_eval, external_eval
    else:
        ids_train, ids_val = train_test_split(ids, test_size = 0.1, random_state = int(trial))
    
        def get_eval_config():
            with open('{}/val/images.json'.format(get_data_dir()), 'r') as f:
                images_eval = json.load(f)
            ids_eval = list(images_eval)
            external_eval = True
            return images_eval, ids_eval, external_eval
        
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

    # Most models are trained on all classes simultaneously
    if SPIRE:
        def counts_batch_cust(y_hat, y):
            return counts_batch(y_hat, y, indices = [0])
    else:
        with open('./Categories.json', 'r') as f:
            cats = json.load(f)
            
        indices = [int(cat['id']) for cat in cats]
        
        def counts_batch_cust(y_hat, y):
            return counts_batch(y_hat, y, indices = indices)
        
    # Setup the data and model; then train
    if INIT and TRANS:
        
        # Setup the dataloaders
        with open('./Models/pretrained/rep.pkl', 'rb') as f:
            rep_pretrained = pickle.load(f)
        
        dataloaders = {}
        for config in [('train', ids_train), ('val', ids_val)]:
            rep_tmp, labels_tmp = load_ids(config[1], rep_pretrained)
            
            rep_tmp = np.array(rep_tmp, dtype = np.float32)
            labels_tmp = np.array(labels_tmp, dtype = np.float32)
            
            rep_tmp = torch.Tensor(rep_tmp)
            labels_tmp = torch.Tensor(labels_tmp)
            
            dataset_tmp = TensorDataset(rep_tmp, labels_tmp)
    
            dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
        
        # Setup the model
        model, _ = get_model(mode = 'transfer', parent = 'pretrained', out_features = 91)
        
        lm = get_lm(model)
        optim_params = lm.parameters()
        lm.cuda()
        
        # Train
        lm = train_model(lm, optim_params, dataloaders, metric_loss, counts_batch_cust, fpr_agg, name = name,
                        lr_init = lr, select_cutoff = select_cutoff, decay_max = decay_max, select_metric_index = select_metric_index,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)
        os.system('rm -rf {}'.format(name))
        
        set_lm(model, lm)
        torch.save(model.state_dict(), '{}.pt'.format(name))
        
    elif INIT and TUNE:

        # Setup the dataloaders
        dataloaders = {}
        for config in [('train', ids_train), ('val', ids_val)]:
            filenames_tmp, labels_tmp = load_ids(config[1], images)
            
            labels_tmp = np.array(labels_tmp, dtype = np.float32)
            
            dataset_tmp = ImageDataset(filenames_tmp, labels_tmp)
    
            dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
            
        # Setup the model
        if HPS:
            parent = './Models/initial-transfer-hps/trial{}/model.pt'.format(trial)
        else:
            parent = './Models/initial-transfer/trial{}/model.pt'.format(trial)
        
        model, optim_params = get_model(mode = 'tune', parent = parent, out_features = 91)
        model.cuda()
        
        # Train
        model = train_model(model, optim_params, dataloaders, metric_loss, counts_batch_cust, fpr_agg, name = name,
                        lr_init = lr, select_cutoff = select_cutoff, decay_max = decay_max, select_metric_index = select_metric_index,
                        mode = mode, mode_param = mode_param, feature_hook = feature_hook)        
        os.system('rm -rf {}'.format(name))
        torch.save(model.state_dict(), '{}.pt'.format(name))
        
        # Save this model's representation of the dataset
        filenames, labels = load_ids(ids, images)
        labels = np.array(labels, dtype = np.float32)
        data = get_representation(model, filenames, labels)
        with open('{}/rep.pkl'.format(model_dir), 'wb') as f:
            pickle.dump(data, f)
            
        # Save this model's predictions on the evaluation data
        out = predict(model, get_eval_config())
        with open('{}/pred.pkl'.format(model_dir), 'wb') as f:
            pickle.dump(out, f)
    
    elif SPIRE and HPS:

        # Setup the model
        parent = './Models/initial-tune-hps/trial{}'.format(trial)
        model, _ = get_model(mode = 'transfer', parent = '{}/model.pt'.format(parent), out_features = 91)
        model.cuda()
        
        # Get the model's representation of the original data
        with open('{}/rep.pkl'.format(parent), 'rb') as f:
            rep_pretrained = pickle.load(f)

        # We adjust the weight of the augmentation for each pair independently
        with open('../0-FindPairs/Pairs.json', 'r') as f:
            pairs = json.load(f)
        
        for pair in pairs: 
            index = get_index(pair)
            
            # Get the upper bound for the sampling probility
            with open('{}/train/splits/{}.json'.format(get_data_dir(), pair), 'r') as f:
                splits = json.load(f)
            
            stats = get_split_stats(splits)
            
            # Load the counterfactuals and get their representations
            cf_data = {}
            for key in ['s_p1', 's_p2']:
                info = stats[key]
                prob = info[0]
                cf_name = info[1]
                
                with open('{}/train/{}/{}/images.json'.format(get_data_dir(), pair, cf_name), 'r') as f:
                    cf_images = json.load(f)
                
                cf_files, cf_labels = load_ids(ids, cf_images, prob = prob)
                cf_labels = np.array(cf_labels, dtype = np.float32)
                
                if len(cf_files) > 0:
                    cf_data[key] = get_representation(model, cf_files, cf_labels)
            
            # Train and eval for each augmentation weight
            # Rather than consider many different weights, we consider fewer weights but get more samples per weight
            scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            num_samples = 5
            out = {}
            for scale in scales:
                scale_ap = 0.0
                for attempt in range(num_samples):
                    
                    # Setup the dataloaders
                    dataloaders = {}
                    for config in [('train', ids_train), ('val', ids_val_all)]:
                        ids_tmp = config[1]

                        # Original Data
                        rep_tmp, labels_tmp = load_ids(ids_tmp, rep_pretrained)
                        
                        rep_tmp = np.array(rep_tmp, dtype = np.float32)
                        labels_tmp = np.array(labels_tmp, dtype = np.float32)
                        
                        x = [rep_tmp]
                        y = [labels_tmp]

                        # Counterfactual data
                        for key in ['s_p1', 's_p2']:
                            if key in cf_data:
                                rep_tmp, labels_tmp = load_ids(ids_tmp, cf_data[key], prob = scale)
                                if len(rep_tmp) > 0:
                                    rep_tmp = np.array(rep_tmp, dtype = np.float32)
                                    labels_tmp = np.array(labels_tmp, dtype = np.float32)

                                    x.append(rep_tmp)
                                    y.append(labels_tmp)

                        # Merge and finish setting up dataloaders
                        x = np.vstack(x)
                        y = np.expand_dims(np.vstack(y)[:, index], 1)

                        x = torch.Tensor(x)
                        y = torch.Tensor(y)

                        dataset_tmp = TensorDataset(x, y)

                        dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = 1024)              

                    # Get the linear model for this class
                    lm = get_lm(model, label_indices = [index])
                    optim_params = lm.parameters()
                    lm.cuda()

                    # Train
                    lm = train_model(lm, optim_params, dataloaders, metric_loss, counts_batch_cust, fpr_agg, name = name,
                                    lr_init = lr, select_cutoff = select_cutoff, decay_max = decay_max, select_metric_index = select_metric_index,
                                    mode = mode, mode_param = mode_param, feature_hook = feature_hook)
                    os.system('rm -rf {}'.format(name))
                    os.system('rm {}.png'.format(name))

                    # Eval
                    rep_eval, _ = load_ids(ids_val_all, rep_pretrained)
                    rep_eval = torch.Tensor(np.array(rep_eval, dtype = np.float32)).cuda()

                    y_hat_eval = torch.sigmoid(lm.forward(rep_eval)).data.cpu().numpy()

                    preds_orig = {}
                    for i, v in enumerate(ids_val_all):
                        preds_orig[v] = y_hat_eval[i]
                    preds = {'orig': preds_orig}
                    
                    info = get_metrics(pair, preds, index = 0, data_split = 'train', max_samples = 500)
                    scale_ap += info['ap']
                
                # Average the estimate for this scale and save
                out[scale] = scale_ap / num_samples
                with open('{}/{}_bap.json'.format(model_dir, pair), 'w') as f:
                    json.dump(out, f)

    elif SPIRE and not HPS:
        
        # Setup the model
        parent = './Models/initial-tune/trial{}'.format(trial)
        model, _ = get_model(mode = 'transfer', parent = '{}/model.pt'.format(parent), out_features = 91)
        model.cuda()
        
        # Get the model's representation of the original data
        with open('{}/rep.pkl'.format(parent), 'rb') as f:
            rep_pretrained = pickle.load(f)

        # We train each class that is part of a SP separately
        with open('./HPS/spire/spire.json', 'r') as f:
            mains = json.load(f)
        
        for main in mains:
            
            index = get_index(main)
            
            # Load the counterfactuals and get their representations
            cf_data = {}
            for info in mains[main]:
                prob = info[0]
                cf_name = info[1]
                
                with open('{}/train/{}/images.json'.format(get_data_dir(), cf_name), 'r') as f:
                    cf_images = json.load(f)
                
                cf_files, cf_labels = load_ids(ids, cf_images, prob = prob)
                cf_labels = np.array(cf_labels, dtype = np.float32)
                
                if len(cf_files) > 0:
                    cf_data[cf_name] = get_representation(model, cf_files, cf_labels)
              
            # Setup the dataloaders
            dataloaders = {}
            for config in [('train', ids_train), ('val', ids_val)]:
                ids_tmp = config[1]

                # Original Data
                rep_tmp, labels_tmp = load_ids(ids_tmp, rep_pretrained)
                
                rep_tmp = np.array(rep_tmp, dtype = np.float32)
                labels_tmp = np.array(labels_tmp, dtype = np.float32)
                
                x = [rep_tmp]
                y = [labels_tmp]

                # Counterfactual data
                for key in cf_data:
                    rep_tmp, labels_tmp = load_ids(ids_tmp, cf_data[key])
                    if len(rep_tmp) > 0:
                        rep_tmp = np.array(rep_tmp, dtype = np.float32)
                        labels_tmp = np.array(labels_tmp, dtype = np.float32)
                        
                        x.append(rep_tmp)
                        y.append(labels_tmp)

                # Merge and finish setting up dataloaders
                x = np.vstack(x)
                y = np.expand_dims(np.vstack(y)[:, index], 1)

                x = torch.Tensor(x)
                y = torch.Tensor(y)

                dataset_tmp = TensorDataset(x, y)

                dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
                
            # Get the linear model for this class
            lm = get_lm(model, label_indices = [index])
            optim_params = lm.parameters()
            lm.cuda()

            # Train
            name_tmp = '{}-{}'.format(name, main)
            lm = train_model(lm, optim_params, dataloaders, metric_loss, counts_batch_cust, fpr_agg, name = name_tmp,
                            lr_init = lr, select_cutoff = select_cutoff, decay_max = decay_max, select_metric_index = select_metric_index,
                            mode = mode, mode_param = mode_param, feature_hook = feature_hook)
            os.system('rm -rf {}'.format(name_tmp))
               
            # Update that class in the main model
            set_lm(model, lm, label_indices = [index])
        
        # Save the final model
        torch.save(model.state_dict(), '{}.pt'.format(name))
        
        # Save this model's predictions on the evaluation data
        out = predict(model, get_eval_config())
        with open('{}/pred.pkl'.format(model_dir), 'wb') as f:
            pickle.dump(out, f)
            
    elif FS:
        
        # Tweak images to match the format expected by the dataloader for FS
        images_orig = {}
        for i in images:
            images_orig[i] = {'orig': images[i]}
        
        # Define when FS uses which features splits and how to weight the examples
        def fs_info(alpha_min = 1):
            with open('../0-FindPairs/Pairs.json', 'r') as f:
                pairs = json.load(f)

            # Create a map from image ID to the indices and weights for the classes whose context should be supressed
            id2info = defaultdict(list)
            for pair in pairs:
                index = get_index(pair)

                # Look at the splits to determine which one needs its context supressed and how much weight to assign it
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
                
                # Apply the lower bound on the weight (this is what is tuned by HPS)
                if alpha < alpha_min:
                    alpha = alpha_min

                # Format the output
                for id in splits[split_suppress]:
                    info = (index, alpha)
                    id2info[id].append(info)
                    
            return id2info
        
        # Either run the HPS search or the final results
        if HPS:
            alpha_list =  [1.0, 10.0, 100.0, 1000.0, 10000.0]
        else:
            # This is the best value accoring to the HPS
            with open('./HPS/fs/fs.json', 'r') as f:
                alpha_list = json.load(f)
        
        out = {}   
        for mode_param in alpha_list:
            
            # Get FS info with this hyper parameter
            id2info = fs_info(alpha_min = mode_param)

            # Setup the dataloaders
            dataloaders = {}
            for config in [('train', ids_train), ('val', ids_val)]:
                filenames_tmp, labels_tmp, contexts_tmp = load_data_fs(config[1], images_orig, id2info)
                dataset_tmp = ImageDataset_FS(filenames_tmp, labels_tmp, contexts_tmp)
                dataloaders[config[0]] = my_dataloader(dataset_tmp, batch_size = batch_size)
            
            # Setup the model
            if HPS:
                parent = './Models/initial-tune-hps/trial{}/model.pt'.format(trial)
            else:
                parent = './Models/initial-tune/trial{}/model.pt'.format(trial)

            model, optim_params = get_model(mode = 'tune', parent = parent, out_features = 91)
            model.cuda()

            feature_hook = get_features(model)
        
            # Train
            model = train_model(model, optim_params, dataloaders, metric_loss, counts_batch_cust, fpr_agg, name = name,
                            lr_init = lr, select_cutoff = select_cutoff, decay_max = decay_max, select_metric_index = select_metric_index,
                            mode = mode, mode_param = mode_param, feature_hook = feature_hook)        
            os.system('rm -rf {}'.format(name))
            
            if HPS:
                os.system('rm {}.png'.format(name))
            else:
                torch.save(model.state_dict(), '{}.pt'.format(name))    

            # Eval
            if HPS:
                pred_eval = predict(model, get_eval_config())

                with open('../0-FindPairs/Pairs.json', 'r') as f:
                    pairs = json.load(f)
                
                v = 0.0
                for pair in pairs:
                    index = get_index(pair)
                    info = get_metrics(pair, pred_eval, index = index, data_split = 'train', max_samples = 500)
                    v += info['ap']
                
                out[mode_param] = v / len(pairs)
                with open('{}/bmap.json'.format(model_dir), 'w') as f:
                    json.dump(out, f)
                
            else:
                # Save this model's predictions on the evaluation data
                out = predict(model, get_eval_config())
                with open('{}/pred.pkl'.format(model_dir), 'wb') as f:
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
