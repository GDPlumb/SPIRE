
import glob
import json
import numpy as np
import os
from subprocess import Popen
import time

# Hyper Parmaeter Search setup
mode = 'initial-tune'
n_trials = 4

print()
print(mode)
print()
time.sleep(3)

# Shared configuration
config = {}
config['main'] = 'bottle'
config['spurious'] = 'person'
config['p_correct'] = 0.90

# Mode specific configuration
config['mode'] = mode

if mode in []:
    config['batch_size'] = -1
else:
    config['batch_size'] = 64

lr_list = [0.001, 0.0001, 0.00001]

if mode in []:
    mp_list = []
else:
    mp_list = [0]

for mp in mp_list:
    config['mode_param'] = mp
    
    for lr in lr_list:
        config['learning_rate'] = lr

        # Create each trial's config file
        for i in range(n_trials):
            config['trial'] = i
            with open('./HPS/{}.json'.format(i), 'w') as f:
                json.dump(config, f)
                
        print('Working on: mp', mp, '& lr', lr)
        
        # Run each trial
        commands = []
        for i in range(n_trials):
            command = 'CUDA_VISIBLE_DEVICES={} python HPS_run.py {}'.format(i, i)
            commands.append(command)
        
        procs = [Popen(i, shell = True) for i in commands]
        for p in procs:
           p.wait()

        # Aggregate and save the results
        base = './HPS/{}-{}-{}/{}/{}-{}/'.format(config['main'], config['spurious'], config['p_correct'], mode, config['mode_param'], config['learning_rate'])

        all_data = []
        for i in range(n_trials):
            file = '{}/trial{}/results.json'.format(base, i)
            with open(file, 'r') as f:
                data = json.load(f)
            all_data.append(data['average'])

        with open('{}/AverageAccuracy.txt'.format(base), 'w') as f:
            print('Mean:', np.mean(all_data), ' & STD:', np.std(all_data), file = f)
