
import json
import numpy as np
import os
from pathlib import Path
import sys

from Heuristics import apply_heuristic
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
                        
if __name__ == '__main__':

    task = sys.argv[1]
    base_location = sys.argv[2]
    
    with open('Configs.json', 'r') as f:
        configs = json.load(f)
        
    configs_all = []
    for configs_worker in configs:
        for config in configs_worker:
            configs_all.append(config)

    for config in configs_all:
    
        mode = config[0]
        n = config[1]
        p = config[2]
        trial = config[3]
    
        save_location = '{}/mode={}/n={}/p={}/trial{}/'.format(base_location, mode, n, p, trial)
        Path(save_location).mkdir(parents = True, exist_ok = True)
        os.chdir(save_location)
        
        if task == 'generate':
            generate(mode, n, p)
        elif task == 'augment':
            generate_augmented(mode)
