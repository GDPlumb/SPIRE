
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import spearmanr
    
if __name__ == '__main__':

    spearman_threshold = 0.5

    back_outer = os.getcwd()
    for pair in glob.glob('./Pairs/*/'):
        os.chdir(pair)
        back = os.getcwd()

        info = pair.split('/')[2].split('-')
        main = info[0]
        spurious = info[1]
        
        # For each training distribution
        all_data = {}
        for p_correct_dir in glob.glob('./0.*/'):
            os.chdir(p_correct_dir)
            back_inner = os.getcwd()
            
            p_correct = p_correct_dir.split('/')[1]
            mode_data = {}
            
            # For each training mode
            for mode_dir in glob.glob('./*'):
                os.chdir(mode_dir)
                
                mode = mode_dir.split('/')[1]
                data = []
            
                # For each trial
                for file in glob.glob('./*/results.json'):
                    with open(file, 'r') as f:
                        data_tmp = json.load(f)
                    data.append(data_tmp)
                
                mode_data[mode] = data
                os.chdir(back_inner)
            
            all_data[p_correct] = mode_data
            os.chdir(back)
            
        # Plot
        p_list = [key for key in all_data]
        p_list = sorted(p_list)
        mode_list = [key for key in all_data[p_list[0]]]
        metric_list = [key for key in all_data[p_list[0]][mode_list[0]][0]]
        
        # Show the effect the co-occurrences have on each metric
        num_plots = len(metric_list)
        
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        count_plots = 1
        
        for metric in metric_list:
            plt.subplot(num_plots, 1, count_plots)
            
            for mode in mode_list:
            
                x_mean = []
                y_mean = []
                x_all = []
                y_all = []
                
                for p in p_list:
                    values = [data[metric] for data in all_data[p][mode]]
                    
                    x_mean.append(p)
                    y_mean.append(np.mean(values))
                    
                    for v in values:
                        x_all.append(p)
                        y_all.append(v)
                    
                    
                plt.plot(x_mean, y_mean, label = mode)
                plt.scatter(x_all, y_all)
                plt.ylabel('Accuracy')
                if metric != 'average':
                    plt.ylim((0, 1))
                plt.xlabel('P(Main | Spurious)')
                plt.title('Distribution: {}'.format(metric))
            plt.legend()
            count_plots += 1
        plt.savefig('Results.png')
        plt.close()
        
        os.chdir(back_outer)
