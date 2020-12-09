
import glob
import json
import numpy as np
import os
from subprocess import Popen
import sys

main = 'bottle'
spurious = 'person'
p_correct = 0.9

modes =['minimal-transfer', 'minimal-tune', 'rrr-tune', 'gs-transfer', 'gs-tune', 'cdep-transfer', 'cdep-tune']
trials = [0, 1, 2, 3]
num_gpus = 4

def get_mp(mode):
    if mode in ['rrr-tune', 'gs-transfer', 'gs-tune', 'cdep-transfer', 'cdep-tune']:
        mp_list = [0.1, 1.0, 10.0, 100.0]
    else:
        mp_list = [0]
    return mp_list
    
def get_lr(mode):
    if 'tune' in mode.split('-'):
        lr_list = [0.0001, 0.0003, 0.00003]
    elif 'transfer' in mode.split('-'):
        lr_list = [0.001, 0.003, 0.0003]
    return lr_list
    
def get_bs(mode):
    if mode in ['cdep-tune']:
        batch_size = 32
    else:
        batch_size = 64
    return batch_size


for mode in modes:

    mp_list = get_mp(mode)
    lr_list = get_lr(mode)
    batch_size = get_bs(mode)
       
    # Generate all of the configurations we want to run
    configs = []
    for mp in mp_list:
        for lr in lr_list:
            for trial in trials:
                config = {}
                # Shared Config
                config['main'] = main
                config['spurious'] = spurious
                config['p_correct'] = p_correct
                config['mode'] = mode
                config['mode_param'] = mp
                config['learning_rate'] = lr
                config['batch_size'] = batch_size
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
        with open('./HPS/{}.json'.format(i), 'w') as f:
            json.dump(configs_worker[i], f)

    # Launch the workers
    commands = []
    for i in range(num_gpus):
        command = 'CUDA_VISIBLE_DEVICES={} python HPS_run.py {}'.format(i, i)
        commands.append(command)

    procs = [Popen(i, shell = True) for i in commands]
    for p in procs:
       p.wait()

    # Cleanup the model files
    os.system('find ./HPS -name model.pt | xargs rm')

    # Aggregate the results
    for mp in mp_list:
        for lr in lr_list:
            dir = './HPS/{}-{}-{}/{}/{}-{}'.format(main, spurious, p_correct, mode, mp, lr, trial)

            all_data = []
            for trial in trials:
                
                with open('{}/trial{}/results.json'.format(dir, trial), 'r') as f:
                    data = json.load(f)
                    all_data.append(data['average'])

            with open('{}/AverageAccuracy.txt'.format(dir), 'w') as f:
                print('Mean:', np.mean(all_data), ' & STD:', np.std(all_data), file = f)
