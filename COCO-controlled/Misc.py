
import numpy as np
import sys

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

###
# Helpers for working with COCO
###

def get_pair(coco, main, spurious):
    
    main = main.replace('+', ' ')
    spurious = spurious.replace('+', ' ')
    
    ids_main = [img['file_name'] for img in coco.get_images_with_cats([main])]
    ids_spurious = [img['file_name'] for img in coco.get_images_with_cats([spurious])]

    both = np.intersect1d(ids_main, ids_spurious)
    just_main = np.setdiff1d(ids_main, ids_spurious)
    just_spurious = np.setdiff1d(ids_spurious, ids_main)
    
    neither = [img['file_name'] for img in coco.get_images_with_cats(None)]
    neither = np.setdiff1d(neither, ids_main)
    neither = np.setdiff1d(neither, ids_spurious)
    
    base_dir = coco.get_base_dir()
    both = ['{}/{}'.format(base_dir, f) for f in both]
    just_main = ['{}/{}'.format(base_dir, f) for f in just_main]
    just_spurious = ['{}/{}'.format(base_dir, f) for f in just_spurious]
    neither = ['{}/{}'.format(base_dir, f) for f in neither]

    return both, just_main, just_spurious, neither
            
def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')
    
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

def load_data_paired(ids, images, name_1, name_2):
    files_1 = []
    labels_1 = []
    files_2 = []
    for id in ids:
        
        # name_1 is not optional: crash if it is missing
        info = images[id][name_1]
        files_1.append(info[0])
        labels_1.append(info[1])
        
        # name_2 is optional:  default to name_1 info if missing
        try:
            info = images[id][name_2]
        except KeyError:
            pass
        
        files_2.append(info[0])

    labels_1 = np.array(labels_1, dtype = np.float32)
    labels_1 = labels_1.reshape((labels_1.shape[0], 1))
    
    return files_1, labels_1, files_2


def load_data_fs(ids, images, splits):
    files = []
    labels = []
    contexts = [] #1 -> spurious (ie, in context), 0 -> no spurious (ie, out of context)
    for id in ids: # For each image
        for key in splits: # Fine which split it came from
            if id in splits[key]:
                files.append(images[id]['orig'][0])
                labels.append(images[id]['orig'][1])
                if key in ['just_main', 'neither']:
                    contexts.append(0)
                elif key in ['both', 'just_spurious']:
                    contexts.append(1)
                else:
                    print('load_data_fs: unexpected key')

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 1))
    
    contexts = np.array(contexts, dtype = np.float32)
    contexts = contexts.reshape((contexts.shape[0], 1))
    
    return files, labels, contexts
