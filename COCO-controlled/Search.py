
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torchvision.models as models

from Misc import get_pair, process_set, id_from_path

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper
from Dataset import ImageDataset, my_dataloader
from FormatData import mask_images_parallel

if __name__ == '__main__':

    # Setup COCO stuff
    coco = COCOWrapper(mode = 'val')
    base_dir = coco.get_base_dir()
    
    imgs = coco.get_images_with_cats(None)
    file2img = {}
    for img in imgs:
        file2img[img['file_name']] = img
        
    # Setup the model
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
    model.cuda()

    # Main Loop - For each pair of objects
    back_outer = os.getcwd()
    for pair in glob.glob('./Pairs/*/'):
        os.chdir(pair)
        back = os.getcwd()
        
        info = pair.split('/')[2].split('-')
        main = info[0]
        spurious = info[1]
        
        # Get relevant data
        both, just_main, just_spurious, neither = get_pair(coco, main, spurious)
        both = both[:100]
        just_main = just_main[:100]
        just_spurious = just_spurious[:100]
        neither = neither[:100]
        
        configs_create = [(both, spurious, 'both-spurious')]
        configs_run = [(both, 'both-spurious')]

        # Create the masked images for the search
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

            imgs = [file2img[f.split('/')[-1]] for f in filenames]
            chosen_id = coco.get_class_id(class_name.replace('+', ' '))
            mask_images_parallel(imgs, coco.coco, base_dir, './', chosen_id = chosen_id)

            os.chdir(back_inner)
        os.chdir(back)
                
        # Run the search
        num_plots = 2 * len(configs_run)
        plot_index = 1
        
        fig = plt.figure(figsize=(15, num_plots * 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
                            
        for config in configs_run:
            filenames = config[0]
            filenames_simple = [f.split('/')[-1] for f in filenames]
            dir_name = config[1]
            
            # For each training distribution
            no2yes_all = {}
            yes2no_all = {}
            for p_correct_dir in glob.glob('./0.*/'):
                os.chdir(p_correct_dir)
                back_inner = os.getcwd()
                
                p_correct = p_correct_dir.split('/')[1]
                
                # For each training mode
                no2yes = {}
                yes2no = {}
                for mode_dir in glob.glob('./*'):
                    os.chdir(mode_dir)
                    
                    mode = mode_dir.split('/')[1]

                    no2yes_tmp = []
                    yes2no_tmp = []
                    
                    # For each trial
                    for file in glob.glob('./*/model.pt'):
                        model.load_state_dict(torch.load(file))
                        model.eval()
                        
                        hat, true, names = process_set(model, filenames, -1, return_value = 'preds', get_names = True)

                        original = {}
                        for i in range(len(hat)):
                            original[id_from_path(names[i])] = 1 * (hat[i] >= 0.5)
                        hat, true, names = process_set(model, filenames_simple, -1, return_value = 'preds', get_names = True, base = '{}/Images-Search/{}'.format(back, dir_name))

                        new = {}
                        for i in range(len(hat)):
                            new[id_from_path(names[i])] = 1 * (hat[i] >= 0.5)

                        matrix = np.zeros((2, 2))
                        for key in original:
                            matrix[original[key], new[key]] += 1

                        no2yes_tmp.append(matrix[0,1] / (matrix[0,0] + matrix[0,1]))
                        yes2no_tmp.append(matrix[1,0] / (matrix[1,0] + matrix[1,1]))
                        
                    no2yes[mode] = no2yes_tmp
                    yes2no[mode] = yes2no_tmp
                    
                    os.chdir(back_inner)
                
                no2yes_all[p_correct] = no2yes
                yes2no_all[p_correct] = yes2no
                
                os.chdir(back)
                    
            p_list = [key for key in no2yes_all]
            p_list = sorted(p_list)
            mode_list = [key for key in no2yes_all[p_list[0]]]
            
            # Plot the results
            plt.subplot(2,1,1)
            plt.suptitle(dir_name)
            
            for mode in mode_list:
                x_mean = []
                y_mean = []
                x_all = []
                y_all = []
                
                for p in p_list:
                    values = no2yes_all[p][mode]
                    
                    x_mean.append(p)
                    y_mean.append(np.mean(values))
                    
                    for v in values:
                        x_all.append(p)
                        y_all.append(v)
                        
                plt.plot(x_mean, y_mean, label = mode)
                plt.scatter(x_all, y_all)
                
                plt.ylabel('New Detection Rate')
                plt.ylim((0, 1))
                plt.xlabel('P(Main | Spurious)')
            plt.legend()
                
            plt.subplot(2,1,2)
            for mode in mode_list:
                x_mean = []
                y_mean = []
                x_all = []
                y_all = []
                
                for p in p_list:
                    values = yes2no_all[p][mode]
                    
                    x_mean.append(p)
                    y_mean.append(np.mean(values))
                    
                    for v in values:
                        x_all.append(p)
                        y_all.append(v)
                        
                plt.plot(x_mean, y_mean, label = mode)
                plt.scatter(x_all, y_all)
                
            plt.ylabel('New Failure Rate')
            plt.ylim((0, 1))
            plt.xlabel('P(Main | Spurious)')
            plt.legend()

            plt.savefig('Search.png')
            plt.close()
        
        os.chdir(back_outer)
