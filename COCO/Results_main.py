
import json
import os
from subprocess import Popen

modes = ['initial-transfer', 'aug-transfer', 'initial-tune', 'aug-tune', 'aug-tp-transfer']
trials = [0, 1, 2, 3]
num_gpus = 4

# Generate all of the configurations we want to run
configs = []
for mode in modes:
    for trial in trials:

        config = {}
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

procs = [Popen(i, shell = True) for i in commands]
for p in procs:
   p.wait()
