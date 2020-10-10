
import json
from multiprocessing import Pool
import numpy as np
import os
import torch
import torchvision.models as models

from COCOWrapper import COCOWrapper
from Dataset import merge_sources, unpack_sources, ImageDataset, my_dataloader
from FormatData import format_spurious
from ModelWrapper import ModelWrapper

def make_key(name_primary, name_spurious):
    return '{}-{}'.format(name_primary, name_spurious).replace(' ', '')

def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')

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

    # Setup the COCO Dataset
    file_dict = merge_sources(['{}/{}{}-info.p'.format(root, mode, year)])
    filenames, labels = unpack_sources(file_dict)
    dataset = ImageDataset(filenames, labels, get_names = True)
    dataloader = my_dataloader(dataset)
    
    # Load a model to use for the search
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    model.load_state_dict(torch.load('./Models/initial-transfer/model_0.pt'))
    model.cuda()
    model.eval()
    
    print('Predicting Original Dataset')
    wrapper = ModelWrapper(model, get_names = True)
    # Get the model predictions on the dataset
    a, b, c = wrapper.predict_dataset(dataloader)

    pred_dict = {}
    dataset_dict = {}
    for i in range(len(a)):
        id = id_from_path(c[i])
        pred_dict[id] = a[i]
        dataset_dict[id] = b[i]
    
    # Get the model predictions on the dataset with the spurious object removed
    # NOTE:  This is the entire dataset, not just the relevant parts
    for name_spurious in names:
        print('Working on: ', name_spurious)
        index_spurious = coco.get_class_id(name_spurious)

        # Divide the images into two sets, one with the spurious object and one without
        images_with_spurious = []
        images_without_spurious = []
        for i in range(len(dataset.filenames)):
            filename = id_from_path(dataset.filenames[i])
            label = dataset.labels[i]
            if label[index_spurious] == 1:
                images_with_spurious.append(filename)
            else:
                images_without_spurious.append(filename)
        
        print('Masking Dataset')
        # Setup a masked dataset for the images with the spurious object
        format_spurious(root, mode, year, name_spurious, use_tmp = True, coco = coco.coco)
        file_dict_spurious = merge_sources(['{}/tmp-info.p'.format(root, mode, year)])
        filenames_spurious, labels_spurious = unpack_sources(file_dict_spurious)
        dataset_spurious = ImageDataset(filenames_spurious, labels_spurious, get_names = True)
        dataloader_spurious = my_dataloader(dataset_spurious)
        
        # Get the model predictions on the images with the spurious object once that object has been removed
        print('Predicting')
        a, b, c = wrapper.predict_dataset(dataloader_spurious)
        pred_without_spurious_dict = {}
        for i in range(len(a)):
            id = id_from_path(c[i])
            pred_without_spurious_dict[id] = a[i]
        
        # Analyze how the spurious object relates to each possible primary object
        print('Analyzing')
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
            for i in range(len(dataset.filenames)):
                filename = id_from_path(dataset.filenames[i])
                label = dataset.labels[i]
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
