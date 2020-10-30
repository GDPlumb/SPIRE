
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot(main, spurious):
    
    save_dir = './Plots/{}-{}'.format(main, spurious)
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
    metric_list = [key for key in all_data[p_list[0]][mode_list[0]][0]]
    
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
    plt.savefig('./Plots/{}-{}/Results.png'.format(main, spurious))
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
    metric_list = [key for key in all_data[p_list[0]][mode_list[0]][0]]
    
    num_plots = 2 * len(metric_list)
    
    fig = plt.figure(figsize=(15, num_plots * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    for metric in metric_list:
        
        for mode in mode_list:
        
            x1_mean = []
            y1_mean = []
            x1_all = []
            y1_all = []
            
            x2_mean = []
            y2_mean = []
            x2_all = []
            y2_all = []
            
            for p in p_list:
                values = [data[metric] for data in all_data[p][mode]]
                
                def no2yes(values):
                    out = []
                    for v in values:
                        out.append(v[0][1] / (v[0][0] + v[0][1]))
                    return out
                    
                def yes2no(values):
                    out = []
                    for v in values:
                        out.append(v[1][0] / (v[1][0] + v[1][1]))
                    return out
                
                v1 = no2yes(values)
                v2 = yes2no(values)
                
                x1_mean.append(p)
                y1_mean.append(np.mean(v1))
                
                for v in v1:
                    x1_all.append(p)
                    y1_all.append(v)
                    
                x2_mean.append(p)
                y2_mean.append(np.mean(v2))
                
                for v in v2:
                    x2_all.append(p)
                    y2_all.append(v)
              
            plt.subplot(num_plots, 1, count_plots)
            plt.plot(x1_mean, y1_mean, label = mode)
            plt.scatter(x1_all, y1_all)
            plt.ylabel('New Detection Rate')
            plt.ylim((0, 1))
            plt.xlabel('P(Main | Spurious)')
            plt.title('Modification: {}'.format(metric))
            plt.legend()
            
            plt.subplot(num_plots, 1, count_plots + 1)
            plt.plot(x2_mean, y2_mean, label = mode)
            plt.scatter(x2_all, y2_all)
            plt.ylabel('New Failure Rate')
            plt.ylim((0, 1))
            plt.xlabel('P(Main | Spurious)')
            plt.title('Modification: {}'.format(metric))
            plt.legend()
            
        count_plots += 2
    plt.savefig('./Plots/{}-{}/Search.png'.format(main, spurious))
    plt.close()    
    
if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]

    plot(main, spurious)
