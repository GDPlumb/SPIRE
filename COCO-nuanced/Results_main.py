
import json
import os
from subprocess import Popen

tuples = ['runway street airplane']
modes = ['initial-transfer', 'initial-tune', 'combined-tune', 'fs-tune', 'added-tune', 'removed-tune']
trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
num_gpus = 4

# Run each tuple in a batch
for tuple in tuples:

    # Generate all of the configurations we want to run
    configs = []

    label1 = tuple.split(' ')[0]
    label2 = tuple.split(' ')[1]
    spurious = tuple.split(' ')[2]
    for mode in modes:
        for trial in trials:
    
            config = {}
            config['label1'] = label1
            config['label2'] = label2
            config['spurious'] = spurious
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

    # Aggregate the results
    os.system('python Analysis.py {} {} {}'.format(label1, label2, spurious))
