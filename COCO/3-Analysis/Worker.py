from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '../')
from Config import get_data_dir

DIR = '../2-Models/Models/{}/trial{}/results.json'

def get_pairs():
    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)
    return pairs
        
#  Get the results for each pair (averaged across trials)
def aggregate_pairs(modes, trials):
    data = {}
    for mode in modes:
        data_mode = defaultdict(list)
        for trial in trials:
            with open(DIR.format(mode, trial), 'r') as f:
                data_tmp = json.load(f)
            for key in data_tmp:
                data_mode[key].append(data_tmp[key])

        for key in data_mode:
            data_tmp = data_mode[key]
            data_mode[key] = '{} ({})'.format(np.round(np.mean(data_tmp), 4), np.round(np.std(data_tmp), 4))

        data[modes[mode]] = data_mode

    # Convert the nested dictionary into a csv
    modes = [mode for mode in data]
    metrics = [key for key in data[modes[0]]]

    # Group the results by pair
    pairs = get_pairs()

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
                tmp.append((name, metric))
        metric_groups[pair] = tmp

    # Save the results
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

        df.to_csv('./Results/{}.csv'.format(group), index = False)               
        
def pair_stats(pair):
    # Load the sizes of each split
    with open('{}/train/splits/{}.json'.format(get_data_dir(), pair), 'r') as f:
        splits = json.load(f)   
    
    # Convert the counts to percentages
    n = 0
    for split in splits:
        splits[split] = len(splits[split])
        n += splits[split]
        
    for split in splits:
        splits[split] /= n

    # Calculate stats
    P_m = splits['both'] + splits['just_main']
    P_s = splits['both'] + splits['just_spurious']
    P_s_given_m = splits['both'] / P_m
    Bias = (P_s_given_m - P_s) / P_s
    Ratio = splits['both'] / splits['just_main']
    
    return P_m, P_s, P_s_given_m, Bias, Ratio

def agg_mean(data, metric):
    return np.mean(data)

def agg_median(data, metric):
    return np.median(data)

def agg_prob(data, metric):
    if metric in ['r-gap', 'h-gap']:
        return np.mean(data < 0)
    else:
        return np.mean(data > 0)

def aggregate_trials(corrected, baseline, trials):
    agg_dict = {'mean': agg_mean, 'median': agg_median, 'prob': agg_prob}
    metrics = ['b-precision', 'b-recall', 'r-gap', 'h-gap']
    
    pairs = get_pairs()

    # Create an output file for each type of aggregation
    for agg_type in agg_dict:
        agg_func = agg_dict[agg_type]

        df = pd.DataFrame()
        df['Mode'] = [corrected[mode] for mode in corrected]

        # Columns are metrics
        for metric in metrics:
            data_column = []
            # Rows are modes
            for mode in corrected:
                data_mode = []
                
                # Report the mean (std) across trials
                for trial in trials:
                    with open('../2-Models/Models/{}/trial{}/results.json'.format(mode, trial), 'r') as f:
                        data_corrected_tmp = json.load(f)
                    with open('../2-Models/Models/{}/trial{}/results.json'.format(baseline, trial), 'r') as f:
                        data_baseline_tmp = json.load(f)

                    # Aggregate over pairs first though
                    data_tmp = []
                    for pair in pairs:
                        name = '{}-{}'.format(pair, metric)
                        v_corrected = data_corrected_tmp[name]
                        v_baseline = data_baseline_tmp[name]
                        if metric in ['r-gap', 'h-gap']:
                            v = np.abs(v_corrected) - np.abs(v_baseline)
                        else:
                            v = v_corrected - v_baseline
                        data_tmp.append(v)
                    data_tmp = np.array(data_tmp)
                    data_mode.append(agg_func(data_tmp, metric))
                data_column.append('{} ({})'.format(np.round(np.mean(data_mode), 3), np.round(np.std(data_mode), 3)))

            df[metric] = data_column

        df.to_csv('./Results/agg_{}.csv'.format(agg_type), index = False)

def df_remove_var_info(df):
    def remove(x):
        return x.split(' ')[0]
    return df.applymap(remove)
    
def show_pairs(corrected, baseline, fontsize = 20): 
    cols = ['Mode', 'both', 'just_main', 'just_spurious', 'neither', 'b-precision', 'b-recall', 'r-gap', 'h-gap']
    modes = [corrected, baseline]
    
    # Group the pairs by Spurious
    pairs = get_pairs()
    groups = defaultdict(list)
    for pair in pairs:
        spurious = pair.split('-')[1]
        groups[spurious].append(pair)

    # Aggregate across pairs
    stats = {}
    stats['P(M)'] = []
    stats['P(S)'] = []
    stats['bias'] = []    
    diffs = defaultdict(list)
    for key in groups:
        group = groups[key]
        for pair in group:                        
            # Calculate basic stats for this pair
            P_m, P_s, P_s_given_m, Bias, Ratio = pair_stats(pair)
            stats['P(M)'].append(P_m)
            stats['P(S)'].append(P_s)
            stats['bias'].append(Bias)

            # For each metric, get the difference between the corrected and baseline models' values (averaged acros trials first)
            df = pd.read_csv('./Results/{}.csv'.format(pair))
            df_tmp = df_remove_var_info(df.loc[df['Mode'].isin(modes), cols].copy())
            for metric in cols[1:]:
                data_tmp = {}
                names = df_tmp['Mode'].values
                values = df_tmp[metric].values
                for i in range(len(names)):
                    if metric in ['r-gap', 'h-gap']:
                        data_tmp[names[i]] = np.abs(float(values[i]))
                    else:
                        data_tmp[names[i]] = float(values[i])
                diffs[metric].append(data_tmp[modes[0]] - data_tmp[modes[1]])

    print('Aggregate Differences:')
    for metric in diffs:
        plt.hist(diffs[metric], bins = 15)
        if metric == 'b-precision':
            plt.title('Change in Balanced Precision')
        elif metric == 'r-gap':
            plt.title('Change in Recall Gap')
        elif metric == 'h-gap':
            plt.title('Change in Hallucination Gap')
        else:
            plt.title(metric)
        plt.show()
        plt.close()
        
        print(metric)
        data = diffs[metric]
        print('mean (std):', np.round(np.mean(data), 3), np.round(np.std(data), 3))
        print('median:', np.round(np.median(data), 3))
        if metric in ['r-gap', 'h-gap']:
            print('prob < 0:',  np.round(np.mean(np.array(data) < 0), 3))
        else:
            print('prob > 0:',  np.round(np.mean(np.array(data) > 0), 3))
        print()
        print()
        
    weight = np.array(stats['P(S)']) * np.array(stats['P(M)'])
     
    plt.scatter(weight, diffs['b-precision'])
    plt.axhline(y=0, linestyle='dashed', c = 'black')
    plt.xlabel('P(Main) * P(Spurious)', fontsize = fontsize)
    plt.ylabel('Change in b-precision', fontsize = fontsize)
    plt.show()
    plt.close()
    
    plt.scatter(stats['bias'], diffs['b-precision'])
    plt.axhline(y=0, linestyle='dashed', c = 'black')
    plt.xlabel('bias', fontsize = fontsize)
    plt.ylabel('Change in b-precision', fontsize = fontsize)
    plt.show()
    plt.close()
    
    plt.scatter(stats['bias'], weight)
    plt.xlabel('bias', fontsize = fontsize)
    plt.ylabel('P(Main) * P(Spurious)', fontsize = fontsize)
    plt.show()
    plt.close()

def make_tex(modes):
    pairs = get_pairs()
    
    meta = []
    for pair in pairs:
        main = pair.split('-')[0].replace('+', ' ')
        spurious = pair.split('-')[1].replace('+', ' ')
        P_m, P_s, P_s_given_m, Bias, Ratio = pair_stats(pair)
        meta.append((main, spurious, P_m, P_s, P_s_given_m, Bias, Ratio))
    meta.sort(key = lambda x: x[5])

    stdout_fileno = sys.stdout

    sys.stdout = open('./Results/all.tex', 'w')
    print('\\centering')
    print('\\resizebox{\\linewidth}{!}{')
    print('\\begin{tabular}{@{}lllllllll@{}}')
    print('\\toprule')
    print('Model& Both& Just Main& Just Spurious& Neither& B-Precision& B-Recall& Recall Gap& Hallucination Gap \\\\ \\midrule')

    for info in meta:

        print('Main: {} & Spurious: {} & P(Main): {} & P(Spurious): {} & P(Spurious \\textbar Main): {} & Bias: {} & Ratio: {} & \\\\'.format(info[0], info[1], info[2], info[3], info[4], info[5], info[6]))

        pair = '{}-{}'.format(info[0], info[1]).replace(' ', '+')
        df = pd.read_csv('./Results/{}.csv'.format(pair))

        df.replace(['initial-tune'], 'Baseline', inplace = True)
        df.replace(['auto-aug'], 'SPIRE', inplace = True)

        for mode in modes:
            row = ''
            for value in df.loc[df['Mode'] == mode].to_numpy()[0]:
                row += '{}&'.format(value)
            row = row[:-1]
            row += '\\\\'
            if mode == 'SPIRE':
                row += ' \\midrule'

            print(row)

    print('\\end{tabular}}')
    sys.stdout = stdout_fileno
    