
import json
import glob
import numpy as np
import os
import pandas as pd
import sys

if __name__ == '__main__':

    label1 = sys.argv[1]
    label2 = sys.argv[2]
    spurious = sys.argv[3]
    main_dir = './Models/{}-{}/{}'.format(label1, label2, spurious)
    
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
    
    df = pd.DataFrame()
    df['Mode'] = modes
    for metric in metrics:
        data_tmp = []
        for mode in modes:
            data_tmp.append(data[mode][metric])
        df[metric] = data_tmp
    
    df.to_csv('./Analysis/{}-{}-{}.csv'.format(label1, label2, spurious))
