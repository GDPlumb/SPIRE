
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torchvision.models as models

from COCODataset import MaskedCOCODataset, COCODataset, my_dataloader
from COCOWrapper import COCOWrapper
from ModelWrapper import ModelWrapper

if __name__ == '__main__':


    torch.multiprocessing.set_sharing_strategy('file_system')

    plot_class = sys.argv[1].replace('-', ' ')
    heuristic_class = sys.argv[2].replace('-', ' ')
    root = sys.argv[3]
    year = sys.argv[4]
        
    num_workers = 8
    
    n_trials = 5

    coco = COCOWrapper(root = root, mode = 'train', year = year)
    index = coco.get_cat_ids(plot_class)[0]

    model_base = './Models'

    model_configs = ['Initial', \
                    'DA/none-[{}]/box-True-default'.format(heuristic_class), \
                    'DA/none-[{}]/box-True-paint'.format(heuristic_class)]

    data_configs = ['val{}-none-[{}]/box-True-default/'.format(year, heuristic_class), \
                    'val{}-none-[{}]/box-True-paint/'.format(year, heuristic_class)]
    
    def my_metrics(model_config, dataloader): # WARNING: this function pulls many variables from the global scope
        p = []
        r = []
        for trial in range(n_trials):
        
            model_file = '{}/{}/model_{}.pt'.format(model_base, model_config, trial)
        
            model.load_state_dict(torch.load(model_file))
            
            y_hat, y_true = wrapper.predict_dataset(dataloader)
            precision, recall = wrapper.metrics(y_hat, y_true)

            p.append(precision[index])
            r.append(recall[index])
            
        return p, r


    # Setup the model and wrapper
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    model.eval()
    model.cuda()
    wrapper = ModelWrapper(model)

    # Format the plot grid
    num_plots = len(data_configs) + 3
    fig = plt.figure(figsize=(5, num_plots * 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count_plots = 1
    
    def update():
        plt.savefig('CompareModels/{}-{}.png'.format(plot_class, heuristic_class))
        print('Finished: ', count_plots - 1)

    # Plot the metrics on the original distribution
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Original Distribution')
    plt.xlabel('MAP')
    plt.ylabel('MAR')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    dataset = COCODataset(root = root, mode = 'val', year = year)
    dataloader = my_dataloader(dataset, num_workers = num_workers)
    
    for model_config in model_configs:
        p = []
        r = []
        for trial in range(n_trials):
        
            model_file = '{}/{}/model_{}.pt'.format(model_base, model_config, trial)
        
            model.load_state_dict(torch.load(model_file))
            
            y_hat, y_true = wrapper.predict_dataset(dataloader)
            precision, recall = wrapper.metrics(y_hat, y_true)
            
            o1, o2 = coco.get_metrics(precision, recall)
            
            p.append(o1)
            r.append(o2)
            
        plt.scatter(p, r, label = model_config)
    plt.legend()
    update()
    
    # Plot the metrics on the natural images split based on whether or not they have the heuristic_class
    imgs_with, imgs_without = coco.split_images_by_cats(cats = [heuristic_class])
    
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Natural Images with {} for detecting {}'.format(heuristic_class, plot_class))
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    dataset = COCODataset(root = root, mode = 'train', year = year, imgIds = imgs_with)
    dataloader = my_dataloader(dataset, num_workers = num_workers)
    for model_config in model_configs:
        p, r = my_metrics(model_config, dataloader)
        plt.scatter(p, r, label = model_config)
    update()
                
    plt.subplot(num_plots, 1, count_plots)
    count_plots += 1
    plt.title('Natural Images without {} for detecting {}'.format(heuristic_class, plot_class))
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    dataset = COCODataset(root = root, mode = 'train', year = year, imgIds = imgs_without)
    dataloader = my_dataloader(dataset, num_workers = num_workers)
    for model_config in model_configs:
        p, r = my_metrics(model_config, dataloader)
        plt.scatter(p, r, label = model_config)
    update()
    
    # Plot the metrics on the altered distributions
    for data_config in data_configs:
    
        plt.subplot(num_plots, 1, count_plots)
        count_plots += 1
        plt.title(data_config)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        dataset = MaskedCOCODataset('./DataAugmentation/{}/labels.p'.format(data_config))
        dataloader = my_dataloader(dataset, num_workers = num_workers)

        for model_config in model_configs:
            p, r = my_metrics(model_config, dataloader)
            plt.scatter(p, r)
        update()
