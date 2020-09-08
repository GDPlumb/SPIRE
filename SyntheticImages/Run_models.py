
import json
import numpy as np
import os
import sys
import torch
import torch.optim as optim

sys.path.insert(0, "../Common/")
from Datasets import StandardDataset, PairedDataset, my_dataloader
from Models import SimpleNet
from Train import train_model

from Heuristics import merge
        
def train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, name, model_load = None,
                lr = 0.001, step_size = 2, gamma = 0.75, num_epochs = 50,
                X_train_aug = None, Y_train_aug = None, X_val_aug = None, Y_val_aug = None, augment = None, # Do we want to use counterfactual data?
                mixup_weight = None, mixup_alpha = 0.1, # Do we want to use mixup as a regularizer?
                rrr_weight = None # Do we want to use RRR as a regularizer?
                ):
                
    if augment == 'standard':
        # Merge the original and heuristic data
        X_train, Y_train = merge(X_train, Y_train, X_train_aug, Y_train_aug)
        X_val, Y_val = merge(X_val, Y_val, X_val_aug, Y_val_aug)
    
    # Configure the dataloaders
    datasets = {}
    if augment == 'paired':
        datasets['train'] = PairedDataset(X_train, Y_train, X_train_aug, Y_train_aug)
        datasets['val'] = PairedDataset(X_val, Y_val, X_val_aug, Y_val_aug)
    else:
        datasets['train'] = StandardDataset(X_train, Y_train)
        datasets['val'] = StandardDataset(X_val, Y_val)
    datasets['test'] = StandardDataset(X_test, Y_test)
    datasets['test_neutral'] = StandardDataset(X_neutral, Y_neutral)

    dataloaders = {}
    for key in datasets:
        dataloaders[key] = my_dataloader(datasets[key])
        
    # Setup the model
    model = SimpleNet()
    if model_load is not None:
        model.load_state_dict(torch.load('{}.pt'.format(model_load)))
    model.train()
    model.cuda()
    
    # Setup the training process
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,
                        num_epochs = num_epochs,
                        mixup_weight = mixup_weight, mixup_alpha = mixup_alpha,
                        rrr_weight = rrr_weight,
                        verbose = False)
    torch.save(model.state_dict(), '{}.pt'.format(name))
    
    # Evaluate the model
    results = {}
    results['train'] = model.accuracy(dataloaders['test'])
    results['neutral'] = model.accuracy(dataloaders['test_neutral'])
    with open('{}.json'.format(name), 'w') as f:
        json.dump(results, f)
    
    return model
    
def train_original():
    # Load the Data
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = np.load(open('data.npy', 'rb'), allow_pickle = True)
    # Train and evaluate
    model = train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, 'model_original')
        
def train_augmented():
    # Load the Data
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = np.load(open('data.npy', 'rb'), allow_pickle = True)
    X_train_aug, Y_train_aug, X_val_aug, Y_val_aug = np.load(open('data_augmented.npy', 'rb'), allow_pickle = True)
    
    # Train and evaluate
    model = train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, 'model_augmented', model_load = 'model_original',
                        X_train_aug = X_train_aug, Y_train_aug = Y_train_aug, X_val_aug = X_val_aug, Y_val_aug = Y_val_aug, augment = 'standard')
 
def train_mixup():
    # Load the Data
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = np.load(open('data.npy', 'rb'), allow_pickle = True)
    X_train_aug, Y_train_aug, X_val_aug, Y_val_aug = np.load(open('data_augmented.npy', 'rb'), allow_pickle = True)
        
    # Train and evaluate
    model = train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, 'model_test', model_load = 'model_original',
                        X_train_aug = X_train_aug, Y_train_aug = Y_train_aug, X_val_aug = X_val_aug, Y_val_aug = Y_val_aug, augment = 'standard',
                        mixup_weight = 0.25, mixup_alpha = 0.2)
 
def train_rrr():
    # Load the Data
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = np.load(open('data.npy', 'rb'), allow_pickle = True)
    X_train_aug, Y_train_aug, X_val_aug, Y_val_aug = np.load(open('data_augmented.npy', 'rb'), allow_pickle = True)
        
    # Train and evaluate
    model = train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, 'model_test',
                        model_load = 'model_original', num_epochs = 25,
                        X_train_aug = X_train_aug, Y_train_aug = Y_train_aug, X_val_aug = X_val_aug, Y_val_aug = Y_val_aug, augment = 'paired',
                        rrr_weight = 0.1)
                        
def train_test():
    train_rrr()
                        
if __name__ == '__main__':

    task = sys.argv[1]
    worker_id = int(sys.argv[2])
    base_location = sys.argv[3]

    with open('Configs.json', 'r') as f:
        configs = json.load(f)
    configs = configs[worker_id]  

    for config in configs:
        
        mode = config[0]
        n = config[1]
        p = config[2]
        trial = config[3]
        
        save_location = '{}/mode={}/n={}/p={}/trial{}/'.format(base_location, mode, n, p, trial)
        os.chdir(save_location)
        
        if task == 'test':
            train_test()
        elif task == 'model_original':
            train_original()
        elif task == 'model_augmented':
            train_augmented()
