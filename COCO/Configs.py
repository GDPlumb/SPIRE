
import json
import os
import random
import sys

root = sys.argv[1]
year = sys.argv[2]
num_workers = int(sys.argv[3])
task = sys.argv[4]

trial_list = [0, 1, 2, 3]

configs = []
for i in range(num_workers):
    configs.append([])

next_worker = 0
for trial in trial_list:

    configs_trial = []

    if task in ['initial-transfer', 'random-transfer', 'initial-tune', 'random-tune', 'random-tune-paint']:
        configs_trial.append([root, year, trial, task])
        
    if task in ['augment-transfer', 'both-transfer']:
        spurious_class = sys.argv[5]
        configs_trial.append([root, year, trial, task, spurious_class])
    
    random.shuffle(configs_trial)
    
    for config in configs_trial:
        configs[next_worker].append(config)
        next_worker = (next_worker + 1) % num_workers
 
with open('Configs.json', 'w') as f:
    json.dump(configs, f)

save_location = './Models/{}'.format(task)
os.system('rm -rf {}'.format(save_location))
os.system('mkdir {}'.format(save_location))
