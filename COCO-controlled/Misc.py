
import numpy as np
import sys

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

###
# Helpers for setting up training data
###

def load_data(ids, images, names):
    files = []
    labels = []
    for id in ids:
        for name in images[id]:
            if name in names:
                files.append(images[id][name][0])
                labels.append(images[id][name][1])
    
    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    return files, labels

def load_data_random(ids, images, splits, splits_data):
    files = []
    labels = []
    for id in ids: # For each image
        for key in splits: # Fine which split it came from
            if id in splits[key]:
                for name in images[id]: # For each version of this image
                    if name in splits_data[key]: # If that version is going to be used
                        if np.random.uniform() <= splits_data[key][name]: # If we sample it, add it
                            files.append(images[id][name][0])
                            labels.append(images[id][name][1])

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    return files, labels
    
def load_data_paired(ids, images, name_cf, aug = False):

    files = []
    labels = []
    files_cf = []
    labels_cf= []
    
    for id in ids:
        info = images[id]['orig']
        files.append(info[0])
        labels.append(info[1])
        
        try:
            info_cf = images[id][name_cf]
            
            files_cf.append(info_cf[0])
            labels_cf.append(info_cf[1])
            
            if aug:
                files.append(info_cf[0])
                labels.append(info_cf[1])
                files_cf.append(info[0])
                labels_cf.append(info[1])
        
        except KeyError: # There is no counterfactual for this image
            files_cf.append(info[0])
            labels_cf.append(info[1])

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    labels_cf = np.array(labels_cf, dtype = np.float32)
    labels_cf = labels_cf.reshape((labels_cf.shape[0], 1))
    
    return files, labels, files_cf, labels_cf

def load_data_fs(ids, images, splits, split_supress, alpha):
    files = []
    labels = []
    contexts = []
    for id in ids: # For each image
        for key in splits: # Fine which split it came from
            if id in splits[key]:
                files.append(images[id]['orig'][0])
                labels.append(images[id]['orig'][1])
                if key in split_supress:
                    contexts.append(alpha)
                else:
                    contexts.append(0)
    
    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    contexts = np.array(contexts, dtype = np.float32)
    contexts = contexts.reshape((contexts.shape[0], 1))
    
    return files, labels, contexts
