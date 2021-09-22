
import json
import numpy as np
import os
from subprocess import Popen
import time

gpu_ids = [0,1,2]#,0,1,2]
num_gpus = len(gpu_ids)

trials = [0,1,2,3,4,5,6,7]

modes = ['fs-tune']

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
    command = 'CUDA_VISIBLE_DEVICES={} python Worker.py {}'.format(gpu_ids[i], i)
    commands.append(command)

procs = []
for i in commands:
    procs.append(Popen(i, shell = True))
    time.sleep(np.random.uniform(4, 6))
    
for p in procs:
   p.wait()

