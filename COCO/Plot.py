
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from COCOWrapper import COCOWrapper

def F1(p, r):
    f = []
    for i in range(len(p)):
        if p[i] == 0 and r[i] == 0:
            f.append(0)
        else:
            f.append(2 * p[i] * r[i] / (p[i] + r[i]))
    return f

if __name__ == '__main__':
    
    root = sys.argv[1]
    year = sys.argv[2]
    main_class = sys.argv[3].replace('-',' ')
    spurious_class = sys.argv[4].replace('-', ' ')
    
    tasks = ['initial-transfer', 'initial-tune', 'random-tune', 'random-tune-paint']

    coco = COCOWrapper(root = root, mode = 'val', year = year)
    
    if main_class == 'main':
    
        # WARNING:  Pulls many things from the global scope
        def partial_plot(datafile):
            p_mean = []
            r_mean = []
            for task in tasks:
                p = []
                r = []
                for file in glob.glob('./Models/{}/model_?.p'.format(task)):
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                        
                    data = data[datafile]
                    o1, o2 = coco.get_metrics(data[0], data[1])
                    p.append(o1)
                    r.append(o2)
                    
                p_mean.append(np.mean(p))
                r_mean.append(np.mean(r))
                                    
                plt.scatter(p, r, label = '{}: F1 {}'.format(task, np.round(np.mean(F1(p, r)), 3)))
            plt.legend()
#            plt.xlim(0,1)
#            plt.ylim(0,1)
            plt.scatter(p_mean, r_mean, marker = '*', s = 500, c = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(p_mean)])
    
        # Format the plot grid
        num_plots = 6
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        count_plots = 1
    
        # Main accuracy plot
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Original Distribution')
        plt.xlabel('MAP')
        plt.ylabel('MAR')
        partial_plot('{}/val{}-info.p'.format(root, year))
        
        # Accuracy without classes
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Original Distribution')
        plt.xlabel('Classless Precision')
        plt.ylabel('Classless Recall')
        
        p_mean = []
        r_mean = []
        for task in tasks:
            p = []
            r = []
            for file in glob.glob('./Models/{}/model_?.p'.format(task)):
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    
                data = data['{}/val{}-info.p'.format(root, year)]
                o1, o2 = coco.get_metrics_classless(1.0 * (data[2] > 0.5), data[3])
                p.append(o1)
                r.append(o2)
                
            p_mean.append(np.mean(p))
            r_mean.append(np.mean(r))
                                
            plt.scatter(p, r, label = '{}: F1 {}'.format(task, np.round(np.mean(F1(p, r)), 3)))
#        plt.xlim(0,1)
#        plt.ylim(0,1)
        plt.legend()
        plt.scatter(p_mean, r_mean, marker = '*', s = 500, c = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(p_mean)])
                
        # Accuracy plots for random removed
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with random masked'.format(spurious_class))
        plt.xlabel('MAP')
        plt.ylabel('MAR')
        partial_plot('{}/val{}-random-info.p'.format(root, year, spurious_class))
                
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with random inpainted'.format(spurious_class))
        plt.xlabel('MAP')
        plt.ylabel('MAR')
        partial_plot('{}/val{}-random-paint-info.p'.format(root, year, spurious_class))
    
         # Accuracy plots for spurious removed
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with {} masked'.format(spurious_class))
        plt.xlabel('MAP')
        plt.ylabel('MAR')
        partial_plot('{}/val{}-{}-info.p'.format(root, year, spurious_class))
                
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with {} inpainted'.format(spurious_class))
        plt.xlabel('MAP')
        plt.ylabel('MAR')
        partial_plot('{}/val{}-{}-paint-info.p'.format(root, year, spurious_class))
        
        plt.savefig('Plot/main.png')

    else:
        index = coco.get_cat_ids(main_class)[0]
    
        # WARNING:  Pulls many things from the global scope
        def partial_plot(datafile):
            p_mean = []
            r_mean = []
            for task in tasks:
                p = []
                r = []
                for file in glob.glob('./Models/{}/model_?.p'.format(task)):
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    
                    data = data[datafile]
                    p.append(data[0][index])
                    r.append(data[1][index])
                        
                p_mean.append(np.mean(p))
                r_mean.append(np.mean(r))
                                    
                plt.scatter(p, r, label = '{}: F1 {}'.format(task, np.round(np.mean(F1(p, r)), 3)))
            
            plt.scatter(p_mean, r_mean, marker = '*', s = 500, c = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(p_mean)])
#            plt.xlim(0,1)
#            plt.ylim(0,1)
            plt.legend()
    

        # Format the plot grid
        num_plots = 5
        fig = plt.figure(figsize=(5, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        count_plots = 1
        
        # Class accuracy
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Dectecting {}'.format(main_class))
        plt.xlabel('precision')
        plt.ylabel('recall')
        partial_plot('{}/val{}-info.p'.format(root, year))
                
        # Class specific accuracy plots for random removed
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with random masked'.format(spurious_class))
        plt.xlabel('precision')
        plt.ylabel('recall')
        partial_plot('{}/val{}-random-info.p'.format(root, year, spurious_class))
                
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with random inpainted'.format(spurious_class))
        plt.xlabel('precision')
        plt.ylabel('recall')
        partial_plot('{}/val{}-random-paint-info.p'.format(root, year, spurious_class))

        # Class specific accuracy plots for spurious removed
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with {} masked'.format(spurious_class))
        plt.xlabel('precision')
        plt.ylabel('recall')
        partial_plot('{}/val{}-{}-info.p'.format(root, year, spurious_class))
                
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title('Counterfactual Images with {} inpainted'.format(spurious_class))
        plt.xlabel('precision')
        plt.ylabel('recall')
        partial_plot('{}/val{}-{}-paint-info.p'.format(root, year, spurious_class))
        
        plt.savefig('Plot/{}-{}.png'.format(main_class, spurious_class))
