
import json
from multiprocessing import Pool
import numpy as np
import os
import torch
import torchvision.models as models

from COCODataset import COCODataset, MaskedCOCOImages, my_dataloader
from COCOWrapper import COCOWrapper
from ModelWrapper import ModelWrapper

def make_key(name_primary, name_spurious):
    return '{}-{}'.format(name_primary, name_spurious).replace(' ', '')

def id_from_path(path):
    return np.int(path.split('/')[-1].split('.')[0].lstrip('0'))

def class_matrix(filenames, dataset_dict, pred_dict, index_primary):
    counts = [0,0,0,0] #[TP, FP, FN, TN]
    for name in filenames:
        label = dataset_dict[name][index_primary]
        pred = pred_dict[name][index_primary]

        if pred == 1:
            if label == 1:
                counts[0] += 1
            else:
                counts[1] += 1
        else:
            if label == 1:
                counts[2] += 1
            else:
                counts[3] += 1
                
    return counts

if __name__ == "__main__":

    os.system('rm -rf Search')
    os.system('mkdir Search')
    
    # Computational Parameters
    workers = 8
    
    # Experimental Configuration
    root = '/home/gregory/Datasets/COCO'
    mode = 'train'
    year = '2017'
    base = '{}/{}{}/'.format(root, mode, year)

    # Setup COCOWrapper
    coco = COCOWrapper(root = root, mode = mode, year = year)

    names = []
    for cat in coco.cats:
        names.append(cat['name'])
    n = len(names)

    # Setup the COCODataset
    dataset = COCODataset(root = root, mode = mode, year = year, get_names = True)
    dataloader = my_dataloader(dataset, num_workers = workers)
    
    # Load a model to use for the search
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    optim_params = model.classifier.parameters()
    model.load_state_dict(torch.load('./Models/Initial/model_0.pt'))
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    # Get the model predictions on the dataset
    a, b, c = wrapper.predict_dataset(dataloader)

    pred_dict = {}
    dataset_dict = {}
    for i in range(len(a)):
        pred_dict[c[i]] = a[i]
        dataset_dict[c[i]] = b[i]
    
    # Get the model predictions on the dataset with the spurious object removed.  NOTE:  This is the entire dataset, not just the relevant parts
    for name_spurious in names:
        index_spurious = coco.get_class_id(name_spurious)

        # Divide the images into two sets, one with the spurious object and one without
        images_with_spurious = []
        images_without_spurious = []
        for item in dataset.ids:
            filename = item[0]
            label = item[1]
            if label[index_spurious] == 1:
                images_with_spurious.append(filename)
            else:
                images_without_spurious.append(filename)
        
        # Setup a masked dataset for the images with the spurious object
        dataset_spurious = MaskedCOCOImages(images_with_spurious, coco.coco, mask_apply = True, mask_classes = [name_spurious], get_names = True)
        dataloader_spurious = my_dataloader(dataset_spurious, num_workers = workers)
        
        # Get the model predictions on the images with the spurious object once that object has been removed
        a, b, c = wrapper.predict_dataset(dataloader_spurious)
        pred_without_spurious_dict = {}
        for i in range(len(a)):
            pred_without_spurious_dict[c[i]] = a[i]
        
        # Analyze how the spurious object relates to each possible primary object
        for name_primary in names:
            index_primary = coco.get_class_id(name_primary)
            key = make_key(name_primary, name_spurious)
            results = {}
            
            # Evaluate the performance gap of the model
            m = class_matrix(images_with_spurious, dataset_dict, pred_dict, index_primary)
            results['matrix_w_spurious'] = m
            m = class_matrix(images_without_spurious, dataset_dict, pred_dict, index_primary)
            results['matrix_wo_spurious'] = m
            
            # Evaluate the strength of the co-occurence between the two objects
            images_with_both = []
            images_with_just_primary = []
            for item in dataset.ids:
                filename = item[0]
                label = item[1]
                if label[index_primary] == 1:
                    if label[index_spurious] == 1:
                        images_with_both.append(filename)
                    else:
                        images_with_just_primary.append(filename)
            results['co-occurence'] = [len(images_with_just_primary), len(images_with_both)]
            
            # Evaluate how often the heuristic changes this prediction
            changed = 0
            count = 0
            # We only consider images that have both objects and that the model detected the primary object
            for filename in images_with_both:
                if pred_dict[filename][index_primary] == 1:
                    count += 1
                    if pred_without_spurious_dict[filename][index_primary] == 0:
                        changed += 1
            results['heuristic_score'] = [changed, count]
            
            
            with open('Search/{}.json'.format(key), 'w') as f:
                json.dump(results, f)
