
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.optim as optim
    
def train_model(model, params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg,
                lr_init = 0.001, decay_phase = 'train', decay_metric = 'loss', decay_min = 0.001, decay_delay = 3, decay_rate = 0.1, decay_max = 2, # Learning rate configuration
                select_metric = 'acc', select_metric_index = 0, select_min = 0.001, select_cutoff = 5, # Model selection configuration
                name = 'history'):
    
    # Setup the learning rate and optimizer
    
    lr = lr_init
    optimizer = optim.Adam(params, lr = lr_init)
    
    # Setup the data logging
    
    loss_history = {}
    loss_history['train'] = []
    loss_history['val'] = []
    
    acc_history = {}
    acc_history['train'] = []
    acc_history['val'] = []
    
    select_history = []
    
    decay_history = []
    
    # Setup the training markers
    
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
                # decay the learning rate
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
            running_acc = []
            running_counts = 0

            for data in dataloaders[phase]:

                x = data[0]
                y = data[1]

                x = x.to('cuda')
                batch_size = x.size(0)
                y = y.to('cuda')
                                        
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(x)
                    loss = metric_loss(pred, y)
                     
                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size
                running_counts += batch_size
                running_acc.append(metric_acc_batch(pred, y))

            epoch_loss = running_loss / running_counts
            epoch_acc_all = metric_acc_agg(counts_list = running_acc)
            epoch_acc = epoch_acc_all[select_metric_index]
            
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc_all)
            
            # check for decay objective progress
            if phase == decay_phase:
                if decay_metric == 'acc':
                    if epoch_acc > decay_value + decay_min:
                        decay_value = epoch_acc
                        decay_time = time
                elif decay_metric == 'loss':
                    if epoch_loss < decay_value - decay_min:
                        decay_value = epoch_loss
                        decay_time = time
                
            # model selection
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
        metrics_names = metric_acc_agg()
        
        fig = plt.figure(figsize=(5, (1 + metrics_num) * 5))
        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        
        x = [i for i in range(time + 1)]
    
        plt.subplot(1 + metrics_num, 1, 1)
        plt.scatter(x, loss_history['train'], label = 'Train')
        plt.scatter(x, loss_history['val'], label = 'Val')
        if decay_metric == 'loss':
            for t in decay_history:
                plt.axvline(t, color = 'black', linestyle = '--')
        if select_metric == 'loss':
            for t in select_history:
                plt.axvline(t, color = 'green', linestyle = '--')
        plt.ylabel('Loss')
        plt.legend()
        
        for i in range(metrics_num):
        
            plt.subplot(1 + metrics_num, 1, 2 + i)
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
            
        plt.savefig('{}.png'.format(name))
        plt.close()
        