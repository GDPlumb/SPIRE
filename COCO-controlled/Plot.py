
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
    all_data = {}
    for p_dir in glob.glob('{}/*'.format(base)):
        p_correct = p_dir.split('/')[-1]
    
        mode_data = {}
        for mode_dir in glob.glob('{}/*'.format(p_dir)):
            mode = mode_dir.split('/')[-1]
            
            data = []
            for file in glob.glob('{}/*/results.json'.format(mode_dir)):
                with open(file, 'r') as f:
                    data_tmp = json.load(f)
                data.append(data_tmp)
            
            mode_data[mode] = data
        all_data[p_correct] = mode_data
        
    # Plot Evaluate Results
    p_list = [key for key in all_data]
    p_list = sorted(p_list)
    mode_list = [key for key in all_data[p_list[0]]]
    mode_list = sorted(mode_list)
    metric_list = [key for key in all_data[p_list[0]][mode_list[0]][0]]
    
    num_plots = len(metric_list)
    
    fig = plt.figure(figsize=(15, num_plots * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    for metric in metric_list:
        plt.subplot(num_plots, 1, count_plots)
        
        for mode in mode_list:
            if (modes_specified is None or mode in modes_specified) and mode not in modes_ignored:
        
                x_mean = []
                y_mean = []
                x_all = []
                y_all = []
                
                for p in p_list:
                    try:
                        values = [data[metric] for data in all_data[p][mode]]
                        
                        x_mean.append(p)
                        y_mean.append(np.mean(values))
                        
                        for v in values:
                            x_all.append(p)
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
    
    # Get Search Results
    all_data = {}
    for p_dir in glob.glob('{}/*'.format(base)):
        p_correct = p_dir.split('/')[-1]
    
        mode_data = {}
        for mode_dir in glob.glob('{}/*'.format(p_dir)):
            mode = mode_dir.split('/')[-1]
            
            data = []
            for file in glob.glob('{}/*/search.json'.format(mode_dir)):
                with open(file, 'r') as f:
                    data_tmp = json.load(f)
                data.append(data_tmp)
            
            mode_data[mode] = data
        all_data[p_correct] = mode_data
    
    # Plot Search Results
    p_list = [key for key in all_data]
    p_list = sorted(p_list)
    mode_list = [key for key in all_data[p_list[0]]]
    mode_list = sorted(mode_list)
    metric_list = [key for key in all_data[p_list[0]][mode_list[0]][0]]
    
    num_plots = len(metric_list)
    
    fig = plt.figure(figsize=(15, num_plots * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    for metric in metric_list:
        
        for mode in mode_list:
            if (modes_specified is None or mode in modes_specified) and mode not in modes_ignored:
        
                x_mean = []
                y_mean = []
                x_all = []
                y_all = []
                
                for p in p_list:
                    try:
                        values = [data[metric] for data in all_data[p][mode]]
                        
                        x_mean.append(p)
                        y_mean.append(np.mean(values))
                        
                        for v in values:
                            x_all.append(p)
                            y_all.append(v)
                            
                    except KeyError:
                        pass
              
                plt.subplot(num_plots, 1, count_plots)
                plt.plot(x_mean, y_mean, label = mode, alpha = 0.5)
                plt.scatter(x_all, y_all, alpha = 0.25)
                plt.ylabel('Probability Prediction Changes')
                if set_ylim:
                    plt.ylim((-1, 1))
                plt.xlabel('P(Main | Spurious)')
                plt.title('Image Split and Counterfactual: {}'.format(metric))
                plt.legend()
            
        count_plots += 1
    plt.savefig('{}/Search.png'.format(save_dir))
    plt.close()    
    
if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]
    plot(main, spurious)
    
    try:
        if sys.argv[3] == 'custom':
            plot(main, spurious, subdir = 'comp-train', modes_specified = ['initial-transfer', 'initial-tune'])
            plot(main, spurious, subdir = 'comp-aug', modes_specified = ['spurious-transfer', 'both-transfer'])
            plot(main, spurious, subdir = 'comp-fill', modes_specified = ['spurious-transfer', 'spurious-paint-transfer', 'spurious-tune', 'spurious-paint-tune'])
            plot(main, spurious, subdir = 'main', modes_specified = ['initial-tune', 'spurious-transfer', 'spurious-tune'])
    except IndexError:
        pass
