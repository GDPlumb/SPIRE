
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import sys

from Worker import get_split_stats

sys.path.insert(0, '../')
from Config import get_data_dir

mode = 'fs'
trials = [0,1,2,3,4,5,6,7]

mode_dir = './HPS/{}'.format(mode)
os.system('rm -rf {}'.format(mode_dir))
os.system('mkdir {}'.format(mode_dir))

if mode == 'spire':
    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)

    spire = defaultdict(list)
    for pair in pairs:
        main = pair.split('-')[0]
        
        try:
            # Get the results for each trial
            # -  Will throw FileNotFoundError if a pair hasn't been started for a trial
            agg = []
            for trial in trials:
                with open('./Models/spire-hps/trial{}/{}_bap.json'.format(trial, pair), 'r') as f:
                    data = json.load(f)
                x = []
                y = []
                for scale in data:
                    x.append(float(scale))
                    y.append(data[scale])   
                plt.plot(x, y, color = 'orange', alpha = 0.25)
                agg.append(y)
            
            # Average the results for each trial
            # - Will throw IndexError if a pair hasn't been finished for a trial
            avg = []
            for i in range(len(x)):
                v = 0
                for trial in trials:
                    v += agg[trial][i]
                v /= len(trials)
                avg.append(v)
            plt.plot(x, avg, color = 'blue', alpha = 0.5)
            
            # Smooth the curve
            avg_smooth = gaussian_filter1d(avg, 1.0)
            plt.plot(x, avg_smooth, color = 'red')
            
            # Find the best scaling factor
            index = np.argmax(avg_smooth[1:]) + 1 # Skip 0
            x_max = x[index]
            y_max = avg_smooth[index]
            plt.scatter(x_max, y_max, marker = '+', s = 72)

            # If this pair has finished for all trials, save
            save_dir = '{}/{}'.format(mode_dir, pair)
            Path(save_dir).mkdir(parents = True, exist_ok = True)

            # Plot the results
            plt.savefig('{}/plot.png'.format(save_dir))
            plt.close()

            # Save the chosen augmentation strategy
            with open('{}/train/splits/{}.json'.format(get_data_dir(), pair), 'r') as f:
                splits = json.load(f)

            stats = get_split_stats(splits)

            out = {}
            for key in ['s_p1', 's_p2']:
                info = stats[key]
                out[key] = (x_max * info[0], info[1])
                spire[main].append((x_max * info[0], '{}/{}'.format(pair, info[1])))

            with open('{}/probs.json'.format(save_dir), 'w') as f:
                json.dump(out, f)

            print(pair)
            print('Sampling Prob: ', out['s_p1'][0])
            print('Boost: ', y_max - avg_smooth[0])
            print()
        except (FileNotFoundError, IndexError):
            break
            
    with open('{}/spire.json'.format(mode_dir), 'w') as f:
        json.dump(spire, f)
        
elif mode == 'fs':
    
    out = defaultdict(list)
    for trial in trials:
        with open('./Models/fs-hps/trial{}/bmap.json'.format(trial), 'r') as f:
            results = json.load(f)
            
        for key in results:
            out[key].append(results[key])
    
    key_best = None
    value_best = -1
    for key in out:
        out[key] = [np.round(np.mean(out[key]), 3), np.round(np.std(out[key]), 3)]
        v = np.mean(out[key])
        if v > value_best:
            value_best = v
            key_best = key
        
    with open('{}/results.json'.format(mode_dir), 'w') as f:
        json.dump(out, f)
    
    with open('{}/fs.json'.format(mode_dir), 'w') as f:
        json.dump([float(key_best)], f)