
import json
import numpy as np
import os
from subprocess import Popen
import time

# 'bottle person', 'bowl person', 'car person', 'chair person', 'cup person', 'dining+table person'
# 'bottle cup', 'bowl cup', 'chair cup'
# 'bottle dining+table', 'bowl dining+table', 'chair dining+table', 'cup dining+table'

HPS = 0 # Are we running Hyper Parameter Selection?
num_gpus = 3

if HPS != 1:
    pairs = ['bottle person', 'bowl person', 'car person', 'chair person', 'cup person', 'dining+table person', 'bottle dining+table', 'chair dining+table']
    modes = ['auto-transfer-ptune', 'simple-transfer-ptune', 'rrr-tune-ptransfer', 'gs-transfer-ptune', 'cdep-transfer-ptune', 'fs-tune-ptune']
    p_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975]
    trials = [0, 1, 2, 3, 4, 5, 6, 7]

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
        command = 'CUDA_VISIBLE_DEVICES={} python Worker.py {} {}'.format(i, HPS, i)
        commands.append(command)

    procs = []
    for i in commands:
        procs.append(Popen(i, shell = True))
        time.sleep(np.random.uniform(4, 6))

    for p in procs:
       p.wait()

else:
    main = 'bottle'
    spurious = 'person'
    p_correct = 0.95

    # 'rrr-tune-ptransfer', 'rrr-tune-ptune'
    # 'cdep-transfer-ptransfer', 'cdep-transfer-ptune', 'cdep-tune-ptransfer', 'cdep-tune-ptune'
    # 'gs-transfer-ptransfer', 'gs-transfer-ptune', 'gs-tune-ptransfer', 'gs-tune-ptune'
    # 'fs-tune-ptransfer', 'fs-tune-ptune'
    # 'auto-transfer-ptransfer', 'auto-transfer-ptune', 'auto-tune-ptransfer', 'auto-tune-ptune'
    modes = []
    trials = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Define the search space
    def get_mp(mode):
        mode_split = mode.split('-')
        if 'rrr' in mode_split:
            mp_list = [0.1, 1.0, 10.0, 100.0]
        elif 'gs' in mode_split:
            mp_list = [0.1, 1.0, 10.0, 100.0, 1000.0]
        elif 'cdep' in mode_split:
            mp_list = [0.1, 1.0, 10.0, 100.0] 
        elif 'fs' in mode_split:
            mp_list = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        else:
            mp_list = [0]
        return mp_list

    def get_lr(mode):
        mode_split = mode.split('-')
        if 'tune' in mode_split:
            lr_list = [0.0001]
        elif 'transfer' in mode_split:
            lr_list = [0.001]
        return lr_list

    def get_bs(mode):
        mode_split = mode.split('-')
        if 'cdep' in mode_split and 'tune' in mode_split:
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
            command = 'CUDA_VISIBLE_DEVICES={} python Worker.py {} {}'.format(i, HPS, i)
            commands.append(command)

        procs = []
        for i in commands:
            procs.append(Popen(i, shell = True))
            time.sleep(np.random.uniform(4, 6))

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
                    print('Mean:', np.mean(all_data), ' & STD:', np.std(all_data), ' & Median:', np.median(all_data), file = f)
