
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from COCOWrapper import COCOWrapper

if __name__ == '__main__':

    print(sys.argv)
    
    root = sys.argv[1]
    year = sys.argv[2]
    main_class = sys.argv[3].replace('-',' ')
    spurious_class = sys.argv[4].replace('-', ' ')
    
    coco = COCOWrapper(root = root, mode = 'val', year = year)
    index = coco.get_cat_ids(main_class)[0]
    
    # WARNING:  Pulls many things from the global scope
    def partial_plot(datafile):
        p_mean = []
        r_mean = []
        for task in tasks:
            p = []
            r = []
            for file in glob.glob('./Models/{}/*.p'.format(task)):
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                
                data = data[datafile]
                p.append(data[0][index])
                r.append(data[1][index])
                    
            p_mean.append(np.mean(p))
            r_mean.append(np.mean(r))
                                
            plt.scatter(p, r, label = task)
        
        plt.scatter(p_mean, r_mean, marker = '*', s = 500, c = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(p_mean)])

    
    tasks = ['initial-transfer', 'random-transfer', 'augment-transfer', 'both-transfer']

    # Format the plot grid
    num_plots = 4
    fig = plt.figure(figsize=(5, num_plots * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    # Main accuracy plot
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Original Distribution')
    plt.xlabel('MAP')
    plt.ylabel('MAR')
    
    p_mean = []
    r_mean = []
    for task in tasks:
        p = []
        r = []
        for file in glob.glob('./Models/{}/*.p'.format(task)):
            with open(file, 'rb') as f:
                data = pickle.load(f)
                
            data = data['{}/val{}-info.p'.format(root, year)]
            o1, o2 = coco.get_metrics(data[0], data[1])
            p.append(o1)
            r.append(o2)
            
        p_mean.append(np.mean(p))
        r_mean.append(np.mean(r))
                            
        plt.scatter(p, r, label = task)
    plt.legend()
    plt.scatter(p_mean, r_mean, marker = '*', s = 500, c = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(p_mean)])
            
    # Class specific accuracy plots
    datafile = '{}/train{}-with-{}-info.p'.format(root, year, spurious_class)
    
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Natural Images with {} for detecting {}'.format(spurious_class, main_class))
    plt.xlabel('precision')
    plt.ylabel('recall')

    partial_plot(datafile)
        
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Natural Images without {} for detecting {}'.format(spurious_class, main_class))
    plt.xlabel('precision')
    plt.ylabel('recall')
    partial_plot('{}/train{}-without-{}-info.p'.format(root, year, spurious_class))
    
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Counterfactual Images with {} removed'.format(spurious_class))
    plt.xlabel('precision')
    plt.ylabel('recall')
    partial_plot('{}/val{}-{}-info.p'.format(root, year, spurious_class))
            
    plt.savefig('Plot/{}-{}.png'.format(main_class, spurious_class))

