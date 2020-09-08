
import json
import random
import sys

config_choice = sys.argv[1]
num_workers = int(sys.argv[2])

if config_choice == 'check':
    mode_list = [1]
    n_list = [15000]
    p_list = [0.95]
    trial_list = [0, 1]
if config_choice == 'test':
    mode_list = [1, 2]
    n_list = [10000, 15000]
    p_list = [0.95]
    trial_list = [0,1,2,3,4]
elif config_choice == 'run':
    modes_list = [1, 2, 3]
    n_list = [5000, 10000, 15000, 20000]
    p_list = [0.5, 0.8, 0.85, 0.9, 0.95, 1.0]
    trial_list = [0,1,2,3,4]
    
configs = []
for i in range(num_workers):
    configs.append([])
 
next_worker = 0
for trial in trial_list:

    configs_trial = []
    for mode in mode_list:
        for n in n_list:
            for p in p_list:
                configs_trial.append([mode, n, p, trial])
                
    random.shuffle(configs_trial)
    
    for config in configs_trial:
        configs[next_worker].append(config)
        next_worker = (next_worker + 1) % num_workers
    
with open('Configs.json', 'w') as f:
    json.dump(configs, f)
