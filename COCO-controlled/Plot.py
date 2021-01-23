
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import warnings
warnings.filterwarnings("ignore")
# Switching back and forth between subplots is going to be deprecated

def plot(main, spurious, subdir = None, modes_specified = None, modes_ignored = [], set_ylim = False):
    
    if subdir is None:
        save_dir = './Plots/{}-{}'.format(main, spurious)
    else:
        save_dir = './Plots/{}-{}/{}'.format(main, spurious, subdir)
    os.system('rm -rf {}'.format(save_dir))
    os.system('mkdir {}'.format(save_dir))
    
    base = '{}/Models/{}-{}'.format(os.getcwd(), main, spurious)
    
    # Get Evaluate Results
    p_list = []
    mode_list = []
    metric_list = []
    all_data = {}
    for p_dir in glob.glob('{}/*'.format(base)):
        p_correct = p_dir.split('/')[-1]
        p_list.append(p_correct)
    
        mode_data = {}
        for mode_dir in glob.glob('{}/*'.format(p_dir)):
            mode = mode_dir.split('/')[-1]
            mode_list.append(mode)
            
            data = []
            for file in glob.glob('{}/*/results.json'.format(mode_dir)):
                with open(file, 'r') as f:
                    data_tmp = json.load(f)
                data.append(data_tmp)
                for key in data_tmp:
                    metric_list.append(key)
            
            mode_data[mode] = data
        all_data[p_correct] = mode_data
        
    # Plot Evaluate Results
    p_list = sorted(list(set(p_list)))
    mode_list = sorted(list(set(mode_list)))
    metric_list = sorted(list(set(metric_list)))
    
    num_plots = len(metric_list)
    
    fig = plt.figure(figsize=(15, num_plots * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    out = {}
    for metric in metric_list:
        out[metric] = {}
        plt.subplot(num_plots, 1, count_plots)
        
        for mode in mode_list:
            if (modes_specified is None or mode in modes_specified) and mode not in modes_ignored:
                out[metric][mode] = {}
        
                x_mean = []
                y_mean = []
                x_all = []
                y_all = []
                
                for p in p_list:
                    try:
                        values = [data[metric] for data in all_data[p][mode]]
                        out[metric][mode][p] = values
                        
                        x_mean.append(float(p))
                        y_mean.append(np.mean(values))
                        
                        for v in values:
                            x_all.append(float(p))
                            y_all.append(v)
                    except KeyError:
                        pass
                        
                plt.plot(x_mean, y_mean, label = mode, alpha = 0.5)
                plt.scatter(x_all, y_all, alpha = 0.25)
                plt.ylabel('Accuracy')
                if set_ylim and metric != 'average':
                    plt.ylim((0, 1))
                plt.xlabel('P(Main | Spurious)')
                plt.title('Image Split: {}'.format(metric))
        plt.legend()
        count_plots += 1
    plt.savefig('{}/Results.png'.format(save_dir))
    plt.close()
    
    with open('{}/Results.json'.format(save_dir), 'w') as f:
        json.dump(out, f)
    
    # Get Search Results
    p_list = []
    mode_list = []
    metric_list = []
    all_data = {}
    for p_dir in glob.glob('{}/*'.format(base)):
        p_correct = p_dir.split('/')[-1]
        p_list.append(p_correct)

        mode_data = {}
        for mode_dir in glob.glob('{}/*'.format(p_dir)):
            mode = mode_dir.split('/')[-1]
            mode_list.append(mode)

            data = []
            for file in glob.glob('{}/*/search.json'.format(mode_dir)):
                with open(file, 'r') as f:
                    data_tmp = json.load(f)
                data.append(data_tmp)
                for key in data_tmp:
                    metric_list.append(key)
                    
            mode_data[mode] = data
        all_data[p_correct] = mode_data
    
    # Plot Search Results
    p_list = sorted(list(set(p_list)))
    mode_list = sorted(list(set(mode_list)))
    metric_list = sorted(list(set(metric_list)))
    
    def get_split(metric):
        chunks = metric.split(' and ')[1].split('-')
        if '+' in chunks[0]:
            return 'add'
        elif 'inverse' in chunks:
            return 'inverse'
        elif 'main' in chunks:
            return 'main'
        elif 'spurious' in chunks:
            return 'spurious'
        else:
            print(metric)
            
    metric_splits = {}
    metric_splits['spurious'] = []
    metric_splits['main'] = []
    metric_splits['add'] = []
    metric_splits['inverse'] = []

    for metric in metric_list:
        metric_splits[get_split(metric)].append(metric)
        
    for split in ['spurious', 'main', 'add']:
    
        metric_list_split = metric_splits[split]
    
        num_plots = len(metric_list_split)
    
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        count_plots = 1
    
        out = {}
        for metric in metric_list_split:
            out[metric] = {}
            
            for mode in mode_list:
                if (modes_specified is None or mode in modes_specified) and mode not in modes_ignored:
                    out[metric][mode] = {}
            
                    x_mean = []
                    y_mean = []
                    x_all = []
                    y_all = []
                    
                    for p in p_list:
                        try:
                            values = [data[metric] for data in all_data[p][mode]]
                            out[metric][mode][p] = values
                            
                            x_mean.append(float(p))
                            y_mean.append(np.mean(values))
                            
                            for v in values:
                                x_all.append(float(p))
                                y_all.append(v)
                                
                        except KeyError:
                            pass
                  
                    plt.subplot(num_plots, 1, count_plots)
                    plt.plot(x_mean, y_mean, label = mode, alpha = 0.5)
                    plt.scatter(x_all, y_all, alpha = 0.25)
                    plt.ylabel('Probability Prediction Changes')
                    if set_ylim:
                        plt.ylim((0, 1))
                    plt.xlabel('P(Main | Spurious)')
                    plt.title('Image Split and Counterfactual: {}'.format(metric))
                    plt.legend()
                
            count_plots += 1
        plt.savefig('{}/Search-{}.png'.format(save_dir, split))
        plt.close()
        
        with open('{}/Search-{}.json'.format(save_dir, split), 'w') as f:
            json.dump(out, f)
    
if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]
    plot(main, spurious)
    
    try:
        if sys.argv[3] == 'custom':
            plot(main, spurious, subdir = 'main', modes_specified = ['initial-tune', 'minimal-tune', 'rrr-tune', 'cdep-tt', 'gs-tt', 'fs-tune'])
    except IndexError:
        pass
