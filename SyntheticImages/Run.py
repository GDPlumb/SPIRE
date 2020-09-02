
import json
import numpy as np
import os
from pathlib import Path
import sys
import torch
import torch.optim as optim

sys.path.insert(0, "../Common/")
from Datasets import StandardDataset, PairedDataset, my_dataloader
from Models import SimpleNet
from Train import train_model

from Heuristics import apply_heuristic, merge
from Sample import load

def generate(mode, n, p, n_neutral = 200):
    if mode == 1:
        from Sample import sample_1 as sample
    elif mode == 2:
        from Sample import sample_2 as sample
    elif mode == 3:
        from Sample import sample_3 as sample
        
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = load(sample, n, p, n_neutral)
    
    with open('data.npy', 'wb') as f:
        np.save(f, [X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test])
        
def train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, name, model_load = None,
                lr = 0.001, step_size = 2, gamma = 0.75, num_epochs = 50,
                X_train_aug = None, Y_train_aug = None, X_val_aug = None, Y_val_aug = None, augment = None, # Do we want to use counterfactual data?
                mixup_weight = None, mixup_alpha = 0.1, # Do we want to use mixup as a regularizer?
                ):
                
    if augment == 'standard':
        # Merge the original and heuristic data
        X_train, Y_train = merge(X_train, Y_train, X_train_aug, Y_train_aug)
        X_val, Y_val = merge(X_val, Y_val, X_val_aug, Y_val_aug)
    
    # Configure the dataloaders
    datasets = {}
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
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs = num_epochs, mixup_weight = mixup_weight, mixup_alpha = mixup_alpha, verbose = False)
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

def generate_augmented(mode):
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = np.load(open('data.npy', 'rb'), allow_pickle = True)
    
    if mode == 1:
        from Heuristics import heuristic_1 as heuristic
    elif mode == 2:
        from Heuristics import heuristic_2 as heuristic
    elif mode == 3:
        from Heuristics import heuristic_3 as heuristic
        
    X_train_aug, Y_train_aug = apply_heuristic(X_train, Y_train, meta_train, heuristic)
    X_val_aug, Y_val_aug = apply_heuristic(X_val, Y_val, meta_val, heuristic)
    
    with open('data_augmented.npy', 'wb') as f:
        np.save(f, [X_train_aug, Y_train_aug, X_val_aug, Y_val_aug])
        
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
    model = train_base(X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, 'model_mixup', model_load = 'model_original',
                        X_train_aug = X_train_aug, Y_train_aug = Y_train_aug, X_val_aug = X_val_aug, Y_val_aug = Y_val_aug, augment = 'standard',
                        mixup_weight = 0.25, mixup_alpha = 0.2)
                        
def train_test():
    train_mixup()
                        
if __name__ == '__main__':

    task = sys.argv[1]
    
    print('')
    print(task)
    print('')
    
    base_location = '/media/gregory/HDD/CounterVision/SyntheticImages/'

    if task == 'test':
        # Remove old test results
        os.chdir(base_location)
        os.system("find -name '*model_test*' | xargs rm")
        
        # Configuration
        modes = [1, 2]
        n_array = [10000, 15000]
        p_array = [0.95]
        trial_array = [0,1,2,3,4]
    else:
        modes = [1,2,3]
        n_array = [5000, 10000, 15000, 20000]
        p_array = [0.5, 0.8, 0.85, 0.9, 0.95, 1.0]
        trial_array = [0,1,2,3,4]

    for trial in trial_array:

        for mode in modes:
            for n in n_array:
                for p in p_array:
                    print(mode, n, p, trial)
                
                    save_location = '{}/mode={}/n={}/p={}/trial{}/'.format(base_location, mode, n, p, trial)
                    Path(save_location).mkdir(parents = True, exist_ok = True)
                    os.chdir(save_location)
                    
                    if task == 'test':
                        train_test()
                    elif task == 'generate':
                        generate(mode, n, p)
                    elif task == 'model_original':
                        train_original()
                    elif task == 'augment':
                        generate_augmented(mode)
                    elif task == 'model_augmented':
                        train_augmented()
