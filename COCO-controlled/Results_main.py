
import json
import os
from subprocess import Popen

#'bottle person' 'bowl person' 'car person' 'chair person' 'cup person' 'dining+table person' 'bottle cup' 'bowl cup' 'chair cup' 'bottle dining+table' 'bowl dining+table' 'chair dining+table' 'cup dining+table'
# pairs = ['bottle person', 'bowl person', 'car person', 'chair person', 'cup person', 'dining+table person']
pairs = ['cup person']
# modes = ['minimal-tune', 'initial-tune']
modes = ['initial-transfer', 'minimal-tune', 'initial-tune']
# p_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975]
p_list = [0.2]
trials = [0, 1, 2, 3]#, 4, 5, 6, 7]
num_gpus = 1

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

procs = [Popen(i, shell = True) for i in commands]
for p in procs:
   p.wait()
