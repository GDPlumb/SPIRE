
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torchvision.models as models

from Misc import get_pair, dist2key

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper
from Dataset import ImageDataset, my_dataloader
from FormatData import mask_images_parallel
from ModelWrapper import ModelWrapper

def process_set(base, files, label):
    files_tmp = []
    labels_tmp = []
    for f in files:
        files_tmp.append('{}/{}'.format(base, f))
        labels_tmp.append(np.array([label], dtype = np.float32))

    labels_tmp = np.array(labels_tmp, dtype = np.float32)

    dataset_tmp = ImageDataset(files_tmp, labels_tmp, get_names = True)

    dataloader_tmp = my_dataloader(dataset_tmp)

    y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)

    return y_hat, y_true, names

def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')
    
if __name__ == '__main__':
    
    root = '/home/gregory/Datasets/COCO'
    year = '2017'

    # Setup COCO stuff
    coco = COCOWrapper(root = root, mode = 'val', year = year)
    base_location = '{}/{}{}'.format(root, 'val', year)
    
    imgs = coco.get_images_with_cats(None)

    file2img = {}
    for img in imgs:
        file2img[img['file_name']] = img
        
    # Setup the model
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
    model.cuda()

    wrapper = ModelWrapper(model, get_names = True)

    # Main Loop - For each pair of objects
    back_main = os.getcwd()
    for pair in glob.glob('./Pairs/*/'):
        info = pair.split('/')[2].split('-')
        main = info[0].replace('+', ' ')
        spurious = info[1].replace('+', ' ')
        
        # Get relevant data
        both, just_main, just_spurious, neither = get_pair(coco, main, spurious)
        both = both[:100]
        just_main = just_main[:100]
        just_spurious = just_spurious[:100]
        neither = neither[:100]
        
        configs_create = [(both, spurious, 'both-spurious')] #, (both, main, 'both-main'), (just_main, main, 'main-main'), (just_spurious, spurious, 'spurious-spurious')]
        configs_run = [(both, 'both-spurious')] #, (both, 'both-main'), (just_main, 'main-main'), (just_spurious, 'spurious-spurious')]

        # Create the masked images for the search
        os.chdir(pair)
        back = os.getcwd()
        
        os.system('rm -rf Images-Search')
        os.system('mkdir Images-Search')
        os.chdir('Images-Search')

        for config in configs_create:
            filenames = config[0]
            class_name = config[1]
            dir_name = config[2]

            os.system('rm -rf {}'.format(dir_name))
            os.system('mkdir {}'.format(dir_name))

            back_inner = os.getcwd()
            os.chdir(dir_name)

            imgs = [file2img[name] for name in filenames]
            chosen_id = coco.get_class_id(class_name)
            mask_images_parallel(imgs, coco.coco, base_location, './', chosen_id = chosen_id)

            os.chdir(back_inner)
            
        os.chdir(back)
                
        # Run the search
        num_plots = 2 * len(configs_run)
        plot_index = 1
        
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
                            
        for config in configs_run:
            filenames = config[0]
            dir_name = config[1]
            
            no2yes = {}
            yes2no = {}
            
            # For each training distribution
            dists = glob.glob('./0.*-0.*/')
            for dist in dists:
                name = dist.split('/')[1]
                
                back_inner = os.getcwd()
                os.chdir(dist)
                
                no2yes_tmp = []
                yes2no_tmp = []

                # For each trial
                for file in glob.glob('./*/model.pt'):
                    model.load_state_dict(torch.load(file))
                    model.eval()

                    hat, true, names = process_set('{}/val{}'.format(root, year), filenames, -1)

                    original = {}
                    for i in range(len(hat)):
                        original[id_from_path(names[i])] = 1 * (hat[i] >= 0.5)
                    hat, true, names = process_set('{}/Images-Search/{}'.format(back_inner, dir_name), filenames, -1)

                    new = {}
                    for i in range(len(hat)):
                        new[id_from_path(names[i])] = 1 * (hat[i] >= 0.5)

                    matrix = np.zeros((2, 2))
                    for key in original:
                        matrix[original[key], new[key]] += 1

                    no2yes_tmp.append(matrix[0,1] / (matrix[0,0] + matrix[0,1]))
                    yes2no_tmp.append(matrix[1,0] / (matrix[1,0] + matrix[1,1]))
                    
                no2yes[name] = no2yes_tmp
                yes2no[name] = yes2no_tmp
                    
                os.chdir(back_inner)
                
            keys = [key for key in no2yes]
            keys = sorted(keys, key = dist2key)
            x = range(len(keys))
                
            # Plot the results
            plt.subplot(2,1,1)
            x_index = 0
            y = []
            y_all = []
            x_all = []
            for key in keys:
                values = no2yes[key]
                y.append(np.mean(values))
                for v in values:
                    y_all.append(v)
                    x_all.append(x_index)
                x_index += 1
            plt.plot(y)
            plt.scatter(x_all, y_all)
            plt.xticks(x, keys)
            plt.ylabel('New Detection Rate')
            plt.title(dir_name)
            
            plt.subplot(2,1,2)
            x_index = 0
            y = []
            y_all = []
            x_all = []
            for key in keys:
                values = yes2no[key]
                y.append(np.mean(values))
                for v in values:
                    y_all.append(v)
                    x_all.append(x_index)
                x_index += 1
            plt.plot(y)
            plt.scatter(x_all, y_all)
            plt.xticks(x, keys)
            plt.ylabel('New Failure Rate')
            
            plt.savefig('Search.png')
            plt.close()
        
        os.chdir(back_main)
