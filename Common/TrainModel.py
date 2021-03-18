
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.optim as optim

from CDEP import cdep_loss
from FS import fs_loss
from GS import gs_loss
from RRR import rrr_loss

    
def train_model(model, params, dataloaders, 
                # Metrics
                metric_loss, metric_acc_batch, metric_acc_agg, 
                # Learning rate configuration
                lr_init = 0.001, decay_phase = 'train', decay_metric = 'loss', decay_min = 0.001, decay_delay = 3, decay_rate = 0.1, decay_max = 2, 
                # Model selection configuration
                select_metric = 'acc', select_metric_index = 0, select_min = 0.001, select_cutoff = 5,
                # Mode configuration
                mode = None, mode_param = None, feature_hook = None,
                # Output configuration
                name = 'history'):
  
    # Mode specific configuration
    mode_split = mode.split('-')
    
    TRANS = 'transfer' in mode_split
    TUNE = 'tune' in mode_split
    
    RRR = 'rrr' in mode_split
    CDEP = 'cdep' in mode_split
    GS = 'gs' in mode_split
    FS = 'fs' in mode_split 
    
    if RRR or CDEP or GS:        
        REG = True
    else:
        REG = False
        
    if TUNE and (RRR or GS):
        INPUT_GRAD = True
        GRAD_DURING_VAL = True
    elif GS and TRANS:
        INPUT_GRAD = False
        GRAD_DURING_VAL = True
    else:
        INPUT_GRAD = False
        GRAD_DURING_VAL = False
    
    if FS:
        rep_avg_running = None
    
    # Setup the learning rate and optimizer
    lr = lr_init
    optimizer = optim.Adam(params, lr = lr_init)
    
    # Setup the data logging
    loss_history = {}
    loss_history['train'] = []
    loss_history['val'] = []
    
    if REG:
        loss_history_reg = {}
        loss_history_reg['train'] = []
        loss_history_reg['val'] = []
    
    acc_history = {}
    acc_history['train'] = []
    acc_history['val'] = []
    
    select_history = []
    
    decay_history = []
    
    # Setup the training tracking
    select_wts = copy.deepcopy(model.state_dict())
    if select_metric == 'acc':
        select_value = 0
    elif select_metric == 'loss':
        select_value = np.inf
    else:
        print('Bad Parameter: select_metric')
        sys.exit(0)
    select_time = 0
    
    if decay_phase not in ['train', 'val']:
        print('Bad Parameter: decay_phase')
        sys.exit(0)
    if decay_metric == 'acc':
        decay_value = 0
    elif decay_metric == 'loss':
        decay_value = np.inf
    else:
        print('Bad Parameter: decay_metric')
        sys.exit(0)
    decay_time = 0
    decay_count = 0
        
    time = -1
    
    # Train
    os.system('rm -rf {}'.format(name))
    os.system('mkdir {}'.format(name))
    
    while True:
        # Check convergence and update the learning rate accordingly
        time += 1
        
        if time - select_time > select_cutoff:
            model.load_state_dict(select_wts)
            return model

        if time - decay_time > decay_delay:
            decay_count += 1
            if decay_count > decay_max:
                model.load_state_dict(select_wts)
                return model
            else:
                # Decay the learning rate
                lr = lr * decay_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                decay_time = time
                decay_history.append(time)

        # Training and validation passes
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            if REG:
                running_loss_reg = 0.0
            running_acc = []
            running_counts = 0

            for data in dataloaders[phase]:

                # Load the data for this batch
                x = data[0]
                y = data[1]

                x = x.to('cuda')
                batch_size = x.size(0)
                y = y.to('cuda')

                if REG:
                    x_prime = data[2]
                    x_prime = x_prime.to('cuda')
                    
                    y_prime = data[3]
                    y_prime = y_prime.to('cuda')
                    
                    if INPUT_GRAD:
                        x.requires_grad = True
                        
                elif FS:
                    c = data[2]
                    c = c.to('cuda')

                # Forward pass
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train' or GRAD_DURING_VAL):
                    pred = model(x)
                    pred_sig = torch.sigmoid(pred)
                    
                    # Main loss
                    if FS:
                        rep = feature_hook.features
                        loss_main, rep_avg_running = fs_loss(rep, rep_avg_running, model, metric_loss, y, c)
                    else:
                        loss_main = metric_loss(pred, y)
                    
                    # Regularizer loss
                    if RRR:
                        loss_reg = rrr_loss(x, x_prime, pred_sig)
                    elif GS and TRANS:
                        rep = feature_hook.features
                        pred_prime = model(x_prime)
                        rep_prime = feature_hook.features
                        loss_reg = gs_loss(rep, rep_prime, y, y_prime, pred_sig)
                    elif GS and TUNE:
                        loss_reg = gs_loss(x, x_prime, y, y_prime, pred_sig)
                    elif CDEP:
                        loss_reg = cdep_loss(x, x_prime, model)
                    
                    # Total loss
                    if REG:
                        loss = loss_main + mode_param * loss_reg
                    else:
                        loss = loss_main
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Calculate batch statistics
                running_loss += loss.item() * batch_size
                if REG:
                    running_loss_reg += loss_reg.item() * batch_size
                running_counts += batch_size
                running_acc.append(metric_acc_batch(pred, y))

            # Calculate epoch statistics
            epoch_loss = running_loss / running_counts
            if REG:
                epoch_loss_reg = running_loss_reg / running_counts
            epoch_acc_all = metric_acc_agg(running_acc)
            epoch_acc = epoch_acc_all[select_metric_index]
            
            # Update the history
            loss_history[phase].append(epoch_loss)
            if REG:
                loss_history_reg[phase].append(epoch_loss_reg)
            acc_history[phase].append(epoch_acc_all)
            
            # Check for decay objective progress
            if phase == decay_phase:
                if decay_metric == 'acc':
                    if epoch_acc > decay_value + decay_min:
                        decay_value = epoch_acc
                        decay_time = time
                elif decay_metric == 'loss':
                    if epoch_loss < decay_value - decay_min:
                        decay_value = epoch_loss
                        decay_time = time
                
            # Model selection
            if phase == 'val':
                if select_metric == 'acc':
                    if epoch_acc > select_value + select_min:
                        select_value = epoch_acc
                        select_time = time
                        select_wts = copy.deepcopy(model.state_dict())
                        select_history.append(time)
                        torch.save(select_wts, '{}/{}.pt'.format(name, time))
                elif select_metric == 'loss':
                    if epoch_loss < select_value - select_min:
                        select_value = epoch_loss
                        select_time = time
                        select_wts = copy.deepcopy(model.state_dict())
                        select_history.append(time)
                        torch.save(select_wts, '{}/{}.pt'.format(name, time))

        # Plot process so far
        metrics_num = len(acc_history['val'][0])
        metrics_names = metric_acc_agg(None)
        
        num_plots = 1 + 1 * REG + metrics_num
        count = 1
        
        fig = plt.figure(figsize=(5, num_plots * 5))
        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        
        x = [i for i in range(time + 1)]
    
        plt.subplot(num_plots, 1, count)
        plt.scatter(x, loss_history['train'], label = 'Train')
        plt.scatter(x, loss_history['val'], label = 'Val')
        if decay_metric == 'loss':
            for t in decay_history:
                plt.axvline(t, color = 'black', linestyle = '--')
        if select_metric == 'loss':
            for t in select_history:
                plt.axvline(t, color = 'green', linestyle = '--')
        plt.ylabel('Loss - Total')
        plt.legend()
        count += 1
        
        if REG:
            plt.subplot(num_plots, 1, count)
            plt.scatter(x, loss_history_reg['train'], label = 'Train')
            plt.scatter(x, loss_history_reg['val'], label = 'Val')
            if decay_metric == 'loss':
                for t in decay_history:
                    plt.axvline(t, color = 'black', linestyle = '--')
            if select_metric == 'loss':
                for t in select_history:
                    plt.axvline(t, color = 'green', linestyle = '--')
            plt.ylabel('Loss - Regularizer')
            plt.legend()
            count += 1
        
        for i in range(metrics_num):
            plt.subplot(num_plots, 1, count)
            plt.scatter(x, [v[i] for v in acc_history['train']], label = 'Train')
            plt.scatter(x, [v[i] for v in acc_history['val']], label = 'Val')
            if i == select_metric_index:
                if decay_metric == 'acc':
                    for t in decay_history:
                        plt.axvline(t, color = 'black', linestyle = '--')
                if select_metric == 'acc':
                    for t in select_history:
                        plt.axvline(t, color = 'green', linestyle = '--')
            plt.xlabel('Time')
            plt.ylabel(metrics_names[i])
            plt.legend()
            count += 1
        plt.savefig('{}.png'.format(name))
        plt.close()
        
###
# Metrics
###

def get_counts(y_hat, y):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_hat[i] == 1:
            if y[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y[i] == 1:
                FN += 1
            else:
                TN += 1
    return [TP, FP, TN, FN]

def counts_batch(y_hat, y, indices = [0]):
    y_hat = y_hat.cpu().data.numpy()
    y_hat = 1 * (y_hat >= 0)
    y = y.cpu().data.numpy()
    out = np.zeros((len(indices), 4))
    for i, index in enumerate(indices):
        out[i, :] = get_counts(y_hat[:, index], y[:, index])
    return out  

def acc_agg(counts_list):
    if counts_list is None:
        return ['Acc']
    else:
        counts_agg = sum(counts_list)
        num_classes = counts_agg.shape[0]
        
        acc = np.zeros((num_classes))
        for i in range(num_classes):
            counts = counts_agg[i, :]
            acc[i] = (counts[0] + counts[2]) / sum(counts)
        
        return [np.mean(acc)]

def fpr_agg(counts_list):
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
