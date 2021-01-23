
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def compare(results, results_type, metric, p_list, modes_comp, mode_base):
    
    data = results[results_type]
    pair_list = [key for key in data]
    
    
    for mode_comp in modes_comp:
    
        x_mean = []
        y_mean = []
        x_all = []
        y_all = []
        
        for p in p_list:
            sum = 0.0
            count = 0
            
            for pair in pair_list:
                try:
                    value_1 = np.mean(data[pair][metric][mode_comp][p])
                    value_2 = np.mean(data[pair][metric][mode_base][p])
                    diff = value_1 - value_2
                    
                    x_all.append(float(p))
                    y_all.append(diff)
                    
                    sum += diff
                    count += 1
                
                except KeyError:
                    print('Missing:', results_type, metric, mode_comp, mode_base, p, pair)
                    pass
            
            x_mean.append(float(p))
            y_mean.append(sum / count)
            
        plt.plot(x_mean, y_mean, label = mode_comp, alpha = 0.5)
        plt.scatter(x_all, y_all, alpha = 0.25)
        plt.hlines(0, min(x_all), max(x_all), color = 'black', linestyle = 'dashed')
        
    plt.title('{}: {}'.format(results_type, metric))
    plt.xlabel('P(Main | Spurious)')
    plt.ylabel('Difference')
    plt.legend()
        
def run(results, p_list, modes_comp, mode_base, title = 'out.png'):

    num_plots = 17
    
    fig = plt.figure(figsize=(4 * 15, 5 * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    plt.suptitle('Baseline is {}'.format(mode_base))

    results_type = 'Results'
    for metric in ['average', 'both', 'just_main', 'just_spurious', 'neither']:
        plt.subplot(4, 5, count_plots)
        count_plots += 1
        compare(results, results_type, metric, p_list, modes_comp, mode_base)

    results_type = 'Search-spurious'
    for metric in ['both and spurious-box', 'both and spurious-pixel-paint', 'just_spurious and spurious-box', 'just_spurious and spurious-pixel-paint']:
        plt.subplot(4, 5, count_plots)
        count_plots += 1
        compare(results, results_type, metric, p_list, modes_comp, mode_base)
    count_plots += 1

    results_type = 'Search-main'
    for metric in ['both and main-box', 'both and main-pixel-paint', 'just_main and main-box', 'just_main and main-pixel-paint']:
        plt.subplot(4, 5, count_plots)
        count_plots += 1
        compare(results, results_type, metric, p_list, modes_comp, mode_base)
    count_plots += 1

    results_type = 'Search-add'
    for metric in ['just_main and just_main+just_spurious', 'just_spurious and just_spurious+just_main', 'neither and neither+just_main', 'neither and neither+just_spurious']:
        plt.subplot(4, 5, count_plots)
        count_plots += 1
        compare(results, results_type, metric, p_list, modes_comp, mode_base)
        
    plt.savefig('./Analysis/{}'.format(title))
    plt.close()


if __name__ == '__main__':

    # Gather all of the data
    base = '{}/Plots'.format(os.getcwd())
    
    results = {} # Type, Pair, Metric, Mode, P - Data
    
    type_list = ['Results', 'Search-spurious', 'Search-main', 'Search-add']
    for results_type in type_list:
        results[results_type] = {}
        for exp_dir in glob.glob('{}/*'.format(base)):
            pair = exp_dir.split('/')[-1]
            try:
                with open('{}/{}.json'.format(exp_dir, results_type), 'r') as f:
                    data = json.load(f)
                results[results_type][pair] = data
            except FileNotFoundError:
                pass
    
    run(results, ['0.025', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '0.9', '0.95', '0.975'], ['minimal-tune', 'rrr-tune', 'gs-tt', 'cdep-tt', 'fs-tune'], 'initial-tune', title = 'main.png')
    run(results, ['0.025', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '0.9', '0.95', '0.975'], ['minimal-tune', 'simple-tune'], 'initial-tune', title = 'aug.png')
