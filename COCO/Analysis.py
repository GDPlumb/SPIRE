
import json
import glob
import numpy as np
import os
import pandas as pd
import sys

if __name__ == '__main__':

    main_dir = './Models'
    
    # Collect the data for each training mode
    data = {}
    for mode_dir in glob.glob('{}/*'.format(main_dir)):
        mode = mode_dir.split('/')[-1]
        
        # Aggregate the data for that mode across the trials
        data_mode = {}
        count = 0
        for trial_dir in glob.glob('{}/trial*'.format(mode_dir)):
            # Include both Accuracy and Search results
            for file in ['results.json', 'search.json']:
                with open('{}/{}'.format(trial_dir, file), 'r') as f:
                    data_tmp = json.load(f)
                for key in data_tmp:
                    if key in data_mode:
                        data_mode[key].append(data_tmp[key])
                    else:
                        data_mode[key] = [data_tmp[key]]
        # We want the average
        for key in data_mode:
            data_tmp = data_mode[key]
            data_mode[key] = '{} ({})'.format(np.round(np.mean(data_tmp), 3), np.round(np.std(data_tmp), 3))
            
        data[mode] = data_mode

    # Convert the nested dictionary into a csv
    modes = [key for key in data]
    modes.sort()
    metrics = [key for key in data[modes[0]]]
    
    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    metric_groups = {}
    metric_groups['avg'] = [('MAP', 'MAP'), ('MAR', 'MAR')]
    
    for pair in pairs:
        n = len(pair)
        main = pair.split('-')[0]
        spurious = pair.split('-')[1]
        n_main = len(main)
        tmp = []
        for metric in metrics:
            if metric[:n] == pair:
                name = metric[n+1:]
                if name[:n_main] == main:
                    name = name[n_main:]
                    if name[0] != '+':
                        name = name[1:]
                name = name.replace(main, 'main')
                name = name.replace(spurious, 'spurious')
                tmp.append((name, metric))
        metric_groups[pair] = tmp
    
    
    for group in metric_groups:
    
        df = pd.DataFrame()
        df['Mode'] = modes
        for info in metric_groups[group]:
            name = info[0]
            metric = info[1]
            data_tmp = []
            for mode in modes:
                data_tmp.append(data[mode][metric])
            df[name] = data_tmp
        
        df.to_csv('./Analysis/{}.csv'.format(group))
