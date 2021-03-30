
from glob import glob
import json
import numpy as np

def load_images(base_dir, cf_types):
    # Get the original images and labels
    with open('{}/images.json'.format(base_dir), 'r') as f:
        data = json.load(f)
        
    images = {}
    for id in data:
        images[id] = {}
        images[id]['orig'] = data[id]
        
    # Get the various counterfactual versions that are available
    for cf_type in cf_types:
        with open('{}/{}/images.json'.format(base_dir, cf_type), 'r') as f:
            data = json.load(f)
        for id in data:
            images[id][cf_type] = data[id]
            
    return images

def load_data(ids, images, img_types, indices_preserve = None):
    files = []
    labels = []
    
    for id in ids:
        for name in images[id]:
            if name in img_types:
                info = images[id][name]
                
                if type(img_types) == dict:
                    if np.random.uniform() <= img_types[name]:
                        files.append(info[0])
                        
                        label = info[1]                        
                        if indices_preserve is not None:
                            for i in range(len(label)):
                                if i not in indices_preserve:
                                    label[i] = 0.0
                        labels.append(label)
                else:
                    files.append(info[0])
                    labels.append(info[1])
    labels = np.array(labels, dtype = np.float32)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, 1)
    
    return files, labels

def load_data_paired(ids, images, cf_types, aug = False):
    files = []
    labels = []
    files_cf = []
    labels_cf = []
    
    for id in ids:
        # Get the original version of the image
        info = images[id]['orig']
        files.append(info[0])
        labels.append(info[1])
        
        # Get the specified counterfactual version of it
        info_cf = None
        for cf_type in cf_types:
            if cf_type in images[id]:
                info_cf = images[id][cf_type]
                break # Each image should have at most 1 counterfactual
        if info_cf is None:
            info_cf = info # There is no counterfactual for this image
        files_cf.append(info_cf[0])
        labels_cf.append(info_cf[1])

        if aug:
            files.append(info_cf[0])
            labels.append(info_cf[1])
            files_cf.append(info[0])
            labels_cf.append(info[1])   

    labels = np.array(labels, dtype = np.float32)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, 1)
        
    labels_cf = np.array(labels_cf, dtype = np.float32)
    if len(labels_cf.shape) == 1:
        labels_cf = np.expand_dims(labels_cf, 1)
    
    return files, labels, files_cf, labels_cf

# id2info[id] = [(index, alpha), ...]
def load_data_fs(ids, images, id2info):
    files = []
    labels = []
    contexts = []
    for id in ids:
        files.append(images[id]['orig'][0])
        label = images[id]['orig'][1]
        labels.append(label)
        # context = 0 -> do not suppress the contex
        # context != 0 -> suppress and use this weight
        if isinstance(label, int):
            context = np.zeros((1))
        elif isinstance(label, list):
            context = np.zeros((len(label)))
        if id in id2info:
            info = id2info[id]
            for pair in info:
                index = pair[0]
                alpha = pair[1]
                context[index] = alpha
        contexts.append(context)
    
    labels = np.array(labels, dtype = np.float32)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, 1)
    
    contexts = np.array(contexts, dtype = np.float32)
    if len(contexts.shape) == 1:
        contexts = np.expand_dims(contexts, 1)

    return files, labels, contexts
