
import json
import numpy as np
import os
from subprocess import Popen
import time

# 'bottle person', 'bowl person', 'car person', 'chair person', 'cup person', 'dining+table person', 'bottle cup', 'bowl cup', 'chair cup', 'bottle dining+table', 'bowl dining+table', 'chair dining+table', 'cup dining+table'
# 'initial-transfer', 'initial-tune', 'minimal-tune', 'simple-tune', 'rrr-tune', 'cdep-tt', 'gs-tt', 'fs-tune'
pairs = ['bottle dining+table',  'chair dining+table']
modes = ['minimal-tune', 'simple-tune', 'rrr-tune', 'cdep-tt', 'gs-tt', 'fs-tune']
p_list = [0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025]
trials = [0, 1, 2, 3]
num_gpus = 4

# Generate all of the configurations we want to run
configs = []
for pair in pairs:
    main = pair.split(' ')[0]
    spurious = pair.split(' ')[1]
    for mode in modes:
        for p in p_list:
            for trial in trials:
        
                config = {}
                config['main'] = main
                config['spurious'] = spurious
                config['p_correct'] = p
                config['mode'] = mode
                config['trial'] = trial
                
                configs.append(config)
        
# Divide the configs among the workers
configs_worker = [[] for i in range(num_gpus)]
next_worker = 0
for config in configs:
    configs_worker[next_worker].append(config)
    next_worker = (next_worker + 1) % num_gpus

# Save the assignments
for i in range(num_gpus):
    with open('./Models/{}.json'.format(i), 'w') as f:
        json.dump(configs_worker[i], f)

# Launch the workers
commands = []
for i in range(num_gpus):
    command = 'CUDA_VISIBLE_DEVICES={} python Results_run.py {}'.format(i, i)
    commands.append(command)

procs = []
for i in commands:
    procs.append(Popen(i, shell = True))
    time.sleep(np.random.uniform(4, 6))

for p in procs:
   p.wait()
