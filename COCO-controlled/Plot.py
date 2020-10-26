
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import spearmanr

def config2key(config):
    info = config.split('-')
    cop_with_main = float(info[0])
    cop_without_main = float(info[1])
    return 1000 * cop_without_main + cop_with_main
    
    
if __name__ == '__main__':

    spearman_threshold = 0.5

    back_main = os.getcwd()

    for pair in glob.glob('./Pairs/*/'):
    
        info = pair.split('/')[2].split('-')
        main = info[0]
        spurious = info[1]
                
        all_data = None
        
        # Collect all of the data for each of the configs for this pair
        os.chdir(pair)
        for config in glob.glob('./*/'):
            
            name = config.split('/')[1]
            info = name.split('-')
            cop_with_main = info[0]
            cop_without_main = info[1]
            
            back = os.getcwd()
            os.chdir(config)
            
            config_data = []
            for file in glob.glob('./*/results.json'):
                with open(file, 'r') as f:
                    data = json.load(f)
                config_data.append(data)
            
            keys = [key for key in config_data[0]]
            
            if all_data is None:
                all_data = {}
                for key in keys:
                    all_data[key] = {}
          
            config_agg = {}
            for key in keys:
                config_agg[key] = []
            for data in config_data:
                for key in keys:
                    config_agg[key].append(data[key])
            
            for key in keys:
                all_data[key][name] = config_agg[key]
                
            os.chdir(back)
        
        # Show the effect the co-occurrences have on each metric
        num_plots = len(all_data)
        
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        count_plots = 1
        
        for key in keys:
            plt.subplot(num_plots, 1, count_plots)
            plt.title(key)

        
            data = all_data[key]
                    
            configs = [config for config in data]
            
            configs = sorted(configs, key = config2key)
            
            y = []
            y_all = []
            x_all = []
            x_index = 0
            for config in configs:
                y.append(np.mean(data[config]))
                
                for value in data[config]:
                    y_all.append(value)
                    x_all.append(x_index)
                x_index += 1
            
            plt.plot(y)
            plt.scatter(x_all, y_all)
            plt.xticks(range(len(configs)), configs)
            count_plots += 1
        plt.savefig('Co-Occurrence.png')
        plt.close()
        
        # Show how each metric compares to the others
        
        data_grouped = {}
        groups = []
        key = keys[0]
        data = all_data[key]
        for i in range(len(configs)):
            config = configs[i]
            for value in data[config]:
                groups.append(i)
        for key in keys:
            tmp = []
            data = all_data[key]

            for config in configs:
                for value in data[config]:
                    tmp.append(value)
                    
            data_grouped[key] = tmp
        
        related_keys = []
        for i in range(len(keys)):
            key1 = keys[i]
            for j in range(i + 1, len(keys)):
                key2 = keys[j]
                
                if np.abs(spearmanr(data_grouped[key1], data_grouped[key2])[0]) > spearman_threshold:
                    related_keys.append((key1, key2))
        
        num_plots = len(related_keys)
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        count_plots = 1
        
        for pair in related_keys:
            plt.subplot(num_plots, 1, count_plots)

            key1 = pair[0]
            key2 = pair[1]
            
            data1 = all_data[key1]
            data2 = all_data[key2]
            
            mean1 = []
            mean2 = []
            for config in configs:
                mean1.append(np.mean(data1[config]))
                mean2.append(np.mean(data2[config]))
                plt.scatter(data1[config], data2[config], label = config)
            
            plt.title('Spearmanr Correlation {}'.format(np.round(spearmanr(data_grouped[key1], data_grouped[key2])[0], 3)))
            plt.xlabel(key1)
            plt.ylabel(key2)
            plt.legend()
            
            plt.scatter(mean1, mean2, marker = '*', s = 500, c = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(mean1)])

            
            count_plots += 1
        
        plt.savefig('MetricComparison.png')
        
        os.chdir(back_main)
