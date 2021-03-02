# This has all been copied from COCO-controlled/Misc.y
# TODO: refactor

import numpy as np

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
    
def load_data_fs(ids, images, splits):

    num_both = len(splits['1s'])
    num_main = len(splits['1ns'])
    
    if num_both >= num_main:
        split_suppress = '1ns'
        alpha = np.sqrt(num_both / num_main)
    else:
        split_suppress = '1s'
        alpha = np.sqrt(num_main / num_both)
    if alpha < 20.0:
        alpha = 20.0
    
    files = []
    labels = []
    contexts = [] #1 -> spurious (ie, in context), 0 -> no spurious (ie, out of context)
    for id in ids: # For each image
        for key in splits: # Find which split it came from
            if id in splits[key]:
                files.append(images[id]['orig'][0])
                labels.append(images[id]['orig'][1])
                if key == split_suppress:
                    contexts.append(alpha)
                else:
                    contexts.append(0.0)

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    contexts = np.array(contexts, dtype = np.float32)
    contexts = contexts.reshape((contexts.shape[0], 1))
    
    return files, labels, contexts

