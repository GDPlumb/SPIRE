# This has all been copied from COCO-controlled/Misc.y
# TODO: refactor

import numpy as np

def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')

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
    files = []
    labels = []
    contexts = [] #1 -> spurious (ie, in context), 0 -> no spurious (ie, out of context)
    for id in ids: # For each image
        for key in splits: # Find which split it came from
            if id in splits[key]:
                files.append(images[id]['orig'][0])
                labels.append(images[id]['orig'][1])
                if key in ['1ns', '0ns']:
                    contexts.append(0)
                elif key in ['1s', '0s']:
                    contexts.append(1)
                else:
                    print('load_data_fs: unexpected key')

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    contexts = np.array(contexts, dtype = np.float32)
    contexts = contexts.reshape((contexts.shape[0], 1))
    
    return files, labels, contexts

